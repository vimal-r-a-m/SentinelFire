# 🔥 SentinelFire — Weakly Supervised Wildfire Verification

This repository contains an end‑to‑end **machine learning systems project** that evaluates whether **spatial–spectral learning using Sentinel‑2 imagery** can improve **second‑stage verification and prioritization** of coarse wildfire alerts produced by MODIS, under **weak supervision**.

The work is framed as a **verification and risk‑ranking problem**, not wildfire prediction or burn‑severity estimation.

---

## 📌 One‑Sentence Summary

> **Given a MODIS fire alert, can high‑resolution multispectral learning better rank how risky that alert looks compared to simple rule‑based spectral indices?**

---

## 🚨 What This Project Is — and Is Not

### ✅ This project **does**
- Treat MODIS fire alerts as **weak supervision**
- Evaluate CNNs as **second‑stage verification models**
- Compare learning against **single‑date NBR heuristics**
- Perform **ablation and disagreement analysis**
- Emphasize **ranking, recall, and robustness**
- Propose a **cost‑aware cascaded deployment strategy**

### ❌ This project **does NOT**
- Predict wildfire ignition
- Estimate burn severity (no dNBR)
- Replace MODIS detection
- Claim pixel‑level ground truth
- Optimize for accuracy, F1, or PR‑AUC

---

## 🗂 Dataset

The project uses the **Sen2Fire** benchmark dataset.

**Inputs**
- Sentinel‑2 multispectral patches (512 × 512, 10 m)
- 12 Sentinel‑2 bands (B1–B12)
- Optional Sentinel‑5P aerosol index

**Labels**
- MODIS MOD14A1 V6.1 daily fire product

**Splits (geographically disjoint)**
- Train: scene1, scene2
- Validation: scene3
- Test: scene4

This split prevents spatial and temporal leakage.

---

## ⚠️ Weak Supervision Assumptions (Critical)

MODIS fire labels are:
- Coarse (1 km resolution)
- Binary
- Temporally misaligned with Sentinel‑2
- Not pixel‑accurate

Consequences:
- Fire occupies a small fraction of each patch
- Many apparent false positives may be label noise
- Some visually burned regions are unlabeled

👉 This motivates **patch‑level classification** and **ranking‑based evaluation**, not pixel‑level segmentation.

---

## 🧠 Problem Formulation

**Task:** Patch‑level binary classification

> “Does this MODIS‑flagged region exhibit fire‑like spatial–spectral evidence in Sentinel‑2 imagery?”

**Why patch‑level?**
- Matches label fidelity
- Avoids circular refinement
- Aligns with MODIS alert semantics

---

## 🧪 Baselines and Models

### Rule‑Based Baseline
- **Single‑date NBR thresholding**
- Represents common GIS heuristics
- Used as a **baseline comparator**, not ground truth

> dNBR is intentionally not used because it requires reliable pre‑ and post‑fire imagery, which is unavailable in this benchmark.

---

### Learned Models

1. **Spectral‑only Logistic Regression**
   - Spatially averaged band values
   - Tests whether spatial context matters

2. **Lightweight CNN (Patch‑Level)**

```
Conv → ReLU → Pool
Conv → ReLU → Global Average Pool → Linear
```

**Why this architecture?**
- Minimal capacity to avoid fitting label noise
- Convolutions capture local spatial patterns
- Global Average Pooling aligns with coarse supervision
- Patch‑level output matches MODIS semantics

---

## 📊 Evaluation Metrics

### Primary Metric — AUROC
- Threshold‑independent
- Robust to class imbalance
- Suitable for **ranking / risk prioritization**

### Reported for interpretation
- Recall (fire): safety‑critical
- Precision (fire): diagnostic only

### Explicitly not used
- Accuracy (misleading under imbalance)
- F1 score (threshold‑sensitive under weak labels)
- PR‑AUC (precision unreliable due to label noise)

---

## 🔍 Disagreement Analysis (Core Contribution)

Rather than relying only on aggregate metrics, the project analyzes **where models disagree**:

| Case | Interpretation |
|----|----|
| CNN ✔ / NBR ✘ | CNN recovers fires missed by brittle index rules |
| CNN ✘ / NBR ✔ | CNN suppresses rule‑based false positives |
| Both ✔ | Easy, obvious fire cases |
| Both ✘ | Clear non‑fire cases |

**Key findings**
- CNN is more robust under smoke, partial burns, and mixed land cover
- NBR frequently triggers on bare soil and seasonal vegetation stress
- Most CNN false negatives correspond to sparse or ambiguous MODIS labels

This analysis explains *why* learning helps, not just *that* it helps.

---

## 🧩 Ablation Studies

The following ablations isolate what contributes to performance:

| Ablation | Observation |
|-------|-------------|
| Spectral‑only | Performance drops → spatial context matters |
| RGB only | Weak performance |
| NIR + SWIR | **Best AUROC (~0.83)** |
| All bands | Performance degrades due to noise |
| + Aerosol | Recall increases, discrimination decreases |

**Conclusion:**  
Targeted spectral selection and spatial context matter more than feature quantity.

---

## 🏗 System Design Perspective

### Evaluation Setup (Used in This Project)

```
MODIS alert
   ↓
Sentinel‑2 patch
   ↓
[NBR]     [CNN]     (parallel evaluation)
```

Used to:
- Understand failure modes
- Justify ML usage
- Avoid evaluation bias

---

### Proposed Production Deployment (Future Work)

```
MODIS alert
   ↓
NBR (cheap rule)
   ↓
NBR‑negative cases
   ↓
CNN (selective verifier)
```

This mirrors **cascaded decision systems** used in spam filtering and fraud detection:
- Rules handle easy cases
- ML corrects systematic failures
- Balanced performance vs compute cost

---

## 📈 Results Summary

- CNN (NIR + SWIR): **AUROC ≈ 0.83**
- Outperforms NBR thresholding and spectral‑only baselines
- Gains come from spatial context and burn‑sensitive bands

---

## 📓 Notebook

All experiments are implemented in:

```
SentinelFire_Weakly_Supervised_Wildfire_Verification.ipynb
```

---

## 🛠 How to Run

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
jupyter notebook SentinelFire_Weakly_Supervised_Wildfire_Verification.ipynb
```

---

## 📌 Resume‑Ready Highlights

- Built a weakly supervised wildfire alert verification system using Sentinel‑2 imagery and MODIS fire products, achieving AUROC ≈ 0.83 on held‑out fire events.
- Demonstrated that lightweight spatial CNNs using NIR–SWIR bands outperform NBR thresholding and spectral‑only baselines under noisy supervision.
- Performed ablation and disagreement analysis to identify where learning improves robustness over rule‑based heuristics.
- Proposed a cost‑aware cascaded deployment strategy where ML selectively corrects rule‑based failures.

---

## ⚖️ Disclaimer

This project is for **research and educational purposes only** and is not an operational wildfire detection system.
