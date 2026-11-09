"""Rewritten program: Adaptive Ridge Regression with Rich Basis Functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def _generate_features(X):
    """
    Generate a rich set of basis functions for each input dimension.
    Includes polynomials, trigonometric functions, and simple non-linear transforms.
    Also includes pairwise interactions.
    """
    n_samples, d = X.shape
    features = []
    
    # Basis functions for each individual dimension
    # We want to avoid too many features to prevent overfitting with N=200
    # Let's use: 1, x, x^2, x^3, sin(pi x), cos(pi x), sin(2 pi x), cos(2 pi x), 1/x, sqrt(x)
    # That's 10 features per dimension.
    
    for j in range(d):
        x = X[:, j]
        
        # Handle potential division by zero or negative sqrt (though inputs are positive per task)
        # Task says inputs in [1, 3], [1, 2], etc., so x > 0.
        
        x_safe = np.maximum(x, 1e-8)
        
        # Polynomials
        features.append(np.ones_like(x))
        features.append(x)
        features.append(x**2)
        features.append(x**3)
        
        # Trig
        features.append(np.sin(np.pi * x))
        features.append(np.cos(np.pi * x))
        features.append(np.sin(2 * np.pi * x))
        features.append(np.cos(2 * np.pi * x))
        
        # Other transforms
        features.append(1.0 / x_safe)
        features.append(np.sqrt(x))
        
    # Interactions (pairwise products of the raw inputs to capture cross terms)
    # If d is large, this explodes. d is max 3.
    # Pairs: (0,1), (0,2), (1,2) for d=3.
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X[:, i] * X[:, j])
            # Maybe also interactions of transformed features? Let's keep it simple first.
            # features.append(np.sin(np.pi * X[:, i]) * np.cos(np.pi * X[:, j])) # Might be too many

    F = np.column_stack(features)
    
    # Ensure no NaNs or Infs
    F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return F

def fit(X_train, y_train):
    """
    Fit a Ridge Regression model with adaptive feature selection/regularization.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin for 300s limit
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    # Generate features
    F_train = _generate_features(X_train)
    
    # If feature generation took too long or failed, fallback to simple poly
    if not np.all(np.isfinite(F_train)):
        F_train = np.column_stack([
            np.ones(len(X_train)),
            X_train,
            X_train**2
        ])
        
    # Select best ridge parameter alpha using Cross Validation
    # We have 200 points. 5-fold CV is reasonable.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid of alphas
    alphas = np.logspace(-4, 4, 9)
    
    best_alpha = 1.0
    best_score = np.inf
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for CV, just use a reasonable default
        best_alpha = 1.0
    else:
        for alpha in alphas:
            mse_scores = []
            for train_idx, val_idx in kf.split(F_train):
                F_tr, F_val = F_train[train_idx], F_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                try:
                    model = Ridge(alpha=alpha)
                    model.fit(F_tr, y_tr)
                    preds = model.predict(F_val)
                    mse = mean_squared_error(y_val, preds)
                    mse_scores.append(mse)
                except Exception:
                    mse_scores.append(np.inf)
            
            mean_mse = np.mean(mse_scores)
            if mean_mse < best_score:
                best_score = mean_mse
                best_alpha = alpha
                
            # Early stop if time is running out
            if time.time() > deadline - 10:
                break

    # Fit final model with best alpha on all training data
    try:
        final_model = Ridge(alpha=best_alpha)
        final_model.fit(F_train, y_train)
    except Exception:
        # Fallback to OLS if Ridge fails for some reason
        final_model = Ridge(alpha=0.0)
        final_model.fit(F_train, y_train)
        
    w = final_model.coef_
    b = final_model.intercept_
    
    # Capture the feature generation logic in the closure
    def predict(X):
        X = np.asarray(X, dtype=float)
        F_test = _generate_features(X)
        # Ensure shapes match
        if F_test.shape[1] != len(w):
             # Fallback if something went wrong with feature consistency (shouldn't happen)
             return np.full(X.shape[0], np.mean(y_train))
        preds = F_test @ w + b
        # Ensure finite output
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
        return preds
    
    return predict
# EVOLVE-BLOCK-END
