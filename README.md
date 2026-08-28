# HealthConnect-Project

---

## 📌 Problem Overview & Business Context

HealthConnect Clinic faces operational inefficiencies and financial losses caused by patients missing scheduled medical appointments. Unattended appointment slots result in:

1. **Wasted Operational Capacity:** Unused medical staff hours and clinical facilities.
2. **Delayed Patient Care:** No-shows prevent other patients on waiting lists from accessing timely healthcare.
3. **Revenue Leakage:** Unutilized time slots directly impact clinic sustainability.

### 🎯 Central Project Question

*Can HealthConnect Clinic leverage historical appointment and patient demographic data to predict, at or before booking time, whether a patient will fail to attend a scheduled appointment?*

By formulating this business problem as a **binary classification model**, the clinic can identify high-risk appointments in advance and deploy targeted intervention strategies (e.g., automated SMS/WhatsApp reminders, direct telephone follow-ups, or strategic slot overbooking).

---

## 📊 Dataset & Exploratory Data Assessment

The dataset (`HealthConnect_Appointment_Data.csv`) contains **5,000 anonymized historical records** spanning 18 distinct variables.

### 1. Data Structure & Summary Statistics

| Metric / Feature | Value / Details |
| :--- | :--- |
| **Total Record Count** | 5,000 records |
| **Unique Patients** | 1,696 patients |
| **Feature Count** | 18 columns (Numeric, Categorical, Datetime) |
| **Primary Key / ID Integrity** | `appointment_id` (0 duplicates) |
| **Target Variable Distribution** (`appointment_outcome`) | **No-Show:** 48.5% (2,423) &nbsp;•&nbsp; **Attended:** 46.3% (2,314) &nbsp;•&nbsp; **Cancelled:** 5.3% (263) |

### 2. Data Quality & Missing Value Assessment

* **`reminder_channel`** (1,366 missing / 27.3%): Structural gap. Cross-tabulation confirms missingness occurs *exclusively* when `reminder_sent == 'No'`. Standardized by imputing `'None'`.
* **`distance_to_clinic_km`** (90 missing / 1.8%): Continuous missing values handled via **median imputation**.
* **`waiting_time_minutes`** (60 missing / 1.2%): Continuous missing values handled via **median imputation**.
* **Logical checks:** Zero instances where `previous_no_shows > previous_appointments` or `booking_lead_days < 0`.

---

## ⚙️ Machine Learning Problem Formulation

### Target Variable Strategy

The raw `appointment_outcome` column has three categories (Attended / No-Show / Cancelled). This is converted into a binary target, `is_no_show`:

* `1` = **No-Show**
* `0` = **Attended**

**Handling cancellations:** The 263 `Cancelled` records (5.3%) represent proactive patient communications rather than unexcused missed slots. These are **excluded** from the baseline classification dataset to prevent target leakage and noise — leaving **4,737 records** (2,423 No-Show / 2,314 Attended) for modelling.

### Proposed Feature Engineering Pipeline

1. **Demographic features:** `age`, `gender`, `age_group`
2. **Behavioral & historical metrics:**
   * `previous_appointments`
   * `previous_no_shows`
   * **Engineered feature:** `historical_no_show_rate` = `previous_no_shows / (previous_appointments + 1)`
3. **Appointment context:**
   * `booking_lead_days` (days between booking and scheduled appointment date)
   * `appointment_day`, `appointment_time` (morning vs. afternoon vs. evening)
   * `distance_to_clinic_km`
4. **Intervention metrics:** `reminder_sent`, `reminder_channel`

---

## 🚀 Initial Modeling Approach & Methodology

1. **Validation strategy:** Implement **GroupKFold cross-validation** grouped on `patient_id`. This prevents data leakage where multiple appointments from the same patient appear in both training and test sets.
2. **Evaluation metrics:**
   * **Primary:** **ROC-AUC** and **PR-AUC** (Precision-Recall AUC, given the near-even class balance and the need for decision thresholding).
   * **Secondary:** **Recall** (sensitivity) for the No-Show class, to capture the maximum number of high-risk patients.

---

## 📌 Key Considerations, Limitations & Next Steps

* **Group leakage risk:** Patient-level repeated visits require group-based splits during model validation.
* **Ethical boundaries:** Predictive outputs must be used strictly for administrative support and patient engagement, avoiding any bias or care deprioritization.
* **Focus for Week 5:**
  * Execute complete data preprocessing and feature encoding pipelines.
  * Train baseline ML models (Logistic Regression, Random Forest, XGBoost).
  * Evaluate feature importances and perform threshold tuning.
