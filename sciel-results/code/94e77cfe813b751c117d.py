"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline
import time


def fit(X_train, y_train):
    X = X_train
    y = y_train
    n = len(X)
    d = X.shape[1]
    
    # All inputs are 1D per the task description
    # Use spline interpolation for smooth 1D data
    # Since data is noiseless, we can use a spline that passes through all points
    
    # Sort by x for spline
    order = np.argsort(X[:, 0])
    x_sorted = X[order, 0]
    y_sorted = y[order]
    
    # Try different spline smoothing parameters
    # s=0 means interpolate exactly (good for noiseless data)
    # We'll use a few candidate splines and pick the best via cross-validation
    
    def make_spline_predict(s):
        try:
            spline = UnivariateSpline(x_sorted, y_sorted, s=s)
            return lambda X_test: spline(X_test[:, 0])
        except Exception:
            return None
    
    # Candidate smoothing values
    s_candidates = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4]
    
    best_predict = None
    best_cv_error = np.inf
    
    # Simple cross-validation: split into train/val
    n_val = max(20, n // 5)
    val_idx = np.random.choice(n, n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    
    x_val = X[val_idx, 0]
    y_val = y[val_idx]
    x_train_cv = X[train_idx, 0]
    y_train_cv = y[train_idx]
    
    for s in s_candidates:
        try:
            spline_cv = UnivariateSpline(x_train_cv, y_train_cv, s=s)
            pred_val = spline_cv(x_val)
            if np.all(np.isfinite(pred_val)):
                cv_err = np.mean((pred_val - y_val) ** 2)
                if cv_err < best_cv_error:
                    best_cv_error = cv_err
                    best_predict = make_spline_predict(s)
        except Exception:
            continue
    
    # If no spline worked, fall back to polynomial fit
    if best_predict is None:
        # Degree 10 polynomial
        def features(X_test):
            x = X_test[:, 0]
            cols = [np.ones(len(x))] + [x ** k for k in range(1, 11)]
            return np.column_stack(cols)
        
        F = features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        best_predict = lambda X_test: features(X_test) @ w
    
    # Final prediction function
    def predict(X_test):
        pred = best_predict(X_test)
        # Ensure finite values
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred
    
    return predict
# EVOLVE-BLOCK-END
