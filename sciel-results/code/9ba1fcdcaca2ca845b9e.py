"""Structural change: Model selection via Cross-Validation between Spline and GPR."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _gpr_predict(X_train, y_train, X_test, length_scale, signal_var):
    """
    Gaussian Process Regression with RBF kernel.
    Assumes y is standardized.
    """
    n = len(X_train)
    # Kernel matrix K(X, X) + noise
    # RBF: exp(-0.5 * (x - x')^2 / l^2)
    # Since 1D, we can compute distances efficiently
    diff = X_train[:, np.newaxis] - X_train[np.newaxis, :]
    K = signal_var * np.exp(-0.5 * diff**2 / length_scale**2)
    
    # Add small jitter for numerical stability
    K += 1e-6 * np.eye(n)
    
    # Factorize
    try:
        L, low = cho_factor(K, lower=True)
        alpha = cho_solve((L, low), y_train)
    except Exception:
        return None, None

    # Predict
    K_s = signal_var * np.exp(-0.5 * (X_test[:, np.newaxis] - X_train[np.newaxis, :])**2 / length_scale**2)
    mu = K_s @ alpha
    
    return mu, None

def _fit_gpr_cv(x, y, n_folds=5):
    """
    Cross-validate GPR to find best hyperparameters.
    Returns (best_mse, best_length_scale, best_signal_var)
    """
    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = len(x) // n_folds
    best_mse = np.inf
    best_l = 1.0
    best_s = 1.0
    
    # Grid search over hyperparameters
    # Length scales: from small (wiggly) to large (smooth)
    # Signal var: usually 1.0 if y is standardized, but let's check
    l_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    s_grid = [0.5, 1.0, 2.0]
    
    for l in l_grid:
        for s in s_grid:
            fold_mses = []
            for i in range(n_folds):
                if _time_left() < 5:
                    return best_mse, best_l, best_s
                
                val_idx = indices[i*fold_size : (i+1)*fold_size]
                train_idx = np.setdiff1d(indices, val_idx)
                
                x_tr = x[train_idx]
                y_tr = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                pred, _ = _gpr_predict(x_tr, y_tr, x_val, l, s)
                if pred is None:
                    continue
                mse = np.mean((pred - y_val)**2)
                fold_mses.append(mse)
            
            if fold_mses:
                avg_mse = np.mean(fold_mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_l = l
                    best_s = s
                    
    return best_mse, best_l, best_s

def _fit_spline_cv(x, y, n_folds=5):
    """
    Cross-validate Cubic Spline.
    Returns (best_mse)
    """
    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = len(x) // n_folds
    fold_mses = []
    
    for i in range(n_folds):
        if _time_left() < 5:
            return np.inf
            
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            cs = CubicSpline(x_tr, y_tr)
            pred = cs(x_val)
            mse = np.mean((pred - y_val)**2)
            fold_mses.append(mse)
        except Exception:
            fold_mses.append(np.inf)
            
    return np.mean(fold_mses)

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Standardize y for GPR stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        def predict(X):
            return np.full(X.shape[0], y_mean)
        return predict
    
    y_norm = (y - y_mean) / y_std
    
    # 1. Evaluate Spline CV
    spline_mse = _fit_spline_cv(x, y_norm)
    
    # 2. Evaluate GPR CV
    gpr_mse, best_l, best_s = _fit_gpr_cv(x, y_norm)
    
    # Decide which model to use
    # If GPR is significantly better, use it. Otherwise, Spline is safer/faster.
    # Note: Spline interpolates, so training MSE is 0. GPR training MSE is > 0.
    # CV MSE is the fair comparison.
    
    use_gpr = gpr_mse < spline_mse
    
    if use_gpr:
        # Fit GPR on all data
        pred_train, _ = _gpr_predict(x, y_norm, x, best_l, best_s)
        if pred_train is None or np.any(~np.isfinite(pred_train)):
            use_gpr = False
        else:
            # Check if GPR is actually good on training data (sanity check)
            train_mse = np.mean((pred_train - y_norm)**2)
            if train_mse > 1.0: # If it's terrible, fallback
                use_gpr = False
    
    if use_gpr:
        def predict(X):
            X_test = X[:, 0]
            pred_norm, _ = _gpr_predict(x, y_norm, X_test, best_l, best_s)
            if pred_norm is None:
                return np.full(X.shape[0], y_mean)
            return pred_norm * y_std + y_mean
    else:
        # Fit Spline on all data
        try:
            cs = CubicSpline(x, y_norm)
            def predict(X):
                X_test = X[:, 0]
                pred_norm = cs(X_test)
                return pred_norm * y_std + y_mean
        except Exception:
            # Fallback to polynomial if spline fails
            try:
                deg = min(10, n-1)
                coeffs = np.polyfit(x, y_norm, deg)
                def predict(X):
                    X_test = X[:, 0]
                    pred_norm = np.polyval(coeffs, X_test)
                    return pred_norm * y_std + y_mean
            except Exception:
                def predict(X):
                    return np.full(X.shape[0], y_mean)
    
    # Final safety wrapper
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.asarray(res, dtype=float)
            res = np.where(np.isfinite(res), res, y_mean)
            return res
        except Exception:
            return np.full(X.shape[0], y_mean)
            
    return safe_predict
# EVOLVE-BLOCK-END
