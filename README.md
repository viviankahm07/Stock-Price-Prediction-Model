# Stock-Price-Prediction-Model

## Overview
The Stock Return Prediction Pipeline is an end-to-end quantitative machine learning system designed to forecast short-horizon equity returns using historical market data. The system models next-period return direction using engineered technical and statistical signals derived from adjusted close prices. It incorporates strict time-series validation, hyperparameter tuning, and leakage prevention to ensure realistic out-of-sample performance estimates. The pipeline mirrors a professional quantitative research workflow, emphasizing reproducibility, modularity, and financial domain rigor.

## Architecture
### Data Acquisition Layer: 
Historical OHLCV data is programmatically downloaded via financial APIs and structured into chronologically ordered, time-indexed datasets. All features are derived from adjusted close prices to account for dividends and stock splits.

### Target Construction Layer: 
Forward returns are computed as percentage changes in adjusted close, including 1-day, 5-day, and 20-day forward returns. For classification tasks, next-day return direction is binarized (positive vs non-positive).

### Feature Engineering Layer: 
Predictive signals are constructed strictly using lagged information to prevent look-ahead bias. Features include:
• 1-day, 5-day, and 20-day trailing returns
• Rolling volatility (standard deviation of returns over 5-day and 20-day windows)
• Simple moving averages (SMA) based on adjusted close (e.g., 5-day, 20-day, 50-day)
• Price-to-moving-average ratios
• Momentum signals
• Rolling mean returns

### Modeling Layer: 
Supervised classification models are trained to predict return direction using expanding-window time-series splits. The modeling framework supports multiple algorithms with cross-validated hyperparameter tuning.

### Evaluation Layer: 
Performance is measured using out-of-sample evaluation across rolling validation folds to simulate realistic forward testing.

## Key Technologies
Python Ecosystem: pandas and numpy for vectorized time-series computation.
Financial Data Retrieval: yfinance for historical equity data.
scikit-learn:
• Logistic Regression
• Random Forest Classifier
• Support Vector Machines (optional)
• TimeSeriesSplit for chronological cross-validation
• GridSearchCV / RandomizedSearchCV for hyperparameter tuning
Matplotlib: Visualization of performance metrics and model diagnostics.

## Features
### Time-Series-Cross Validation: 
Implements expanding-window cross-validation using TimeSeriesSplit to prevent data leakage and maintain temporal ordering.

### Hyperparameter Tuning:
Uses structured search procedures (GridSearchCV or RandomizedSearchCV) to optimize model parameters such as:
• Regularization strength (C) for Logistic Regression
• Number of estimators, max depth, and minimum samples split for Random Forest
• Kernel parameters for SVM

### Multi-Horizon Return Engineering: 
Explicit modeling of 1-day, 5-day, and 20-day return structures to capture short-term momentum and medium-term trends.

### Volatility Modeling:
Rolling standard deviation of adjusted close returns captures local market uncertainty regimes.

### Trend Signals: 
Moving averages and relative price deviations provide interpretable momentum-based predictors.

### Performance Metrics:
Models are evaluated using:
• Accuracy
• Precision
• Recall
• F1-score
• ROC-AUC
• Confusion Matrix

These metrics are computed on strictly out-of-sample validation folds to assess predictive stability rather than in-sample fit.

## Engineering Highlights
Strict separation of feature windows and forward returns eliminates look-ahead bias.
Implements multi-horizon signal construction aligned with quantitative trading conventions.
Applies hyperparameter optimization within a time-series cross-validation framework.
Demonstrates understanding of non-i.i.d financial data and regime sensitivity.
Modular notebook structure separates data ingestion, feature construction, and modeling logic for reproducibility.

## Summary
This project replicates the workflow of a quantitative research pipeline, from adjusted close preprocessing to multi-horizon signal engineering and hyperparameter-optimized model training. It demonstrates the ability to design financially informed features, implement time-aware validation strategies, and rigorously evaluate predictive models in noisy, non-stationary markets. The system reflects both machine learning technical depth and domain-specific quantitative reasoning, aligning closely with the standards of quantitative trading and applied ML roles.
