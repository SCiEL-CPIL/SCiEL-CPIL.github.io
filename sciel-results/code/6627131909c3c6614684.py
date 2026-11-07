"""Structural change: Robust 1D Regression with Sorted Data and Model Selection (Poly vs Spline)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _make_poly_matrix(x, deg):
    """Create polynomial feature matrix with bias."""
    x = x.reshape(-1, 1)
    # x^0, x^1, ..., x^deg
    cols = [np.ones_like(x)]
    for i in range(1, deg + 1):
        cols.append(x ** i)
    return np.hstack(cols)

def _cv_score(x, y, model_type, param, n_folds=5):
    """
    Calculate CV MSE for a given model type and parameter.
    model_type: 'poly' or 'spline'
    param: degree for poly
    """
    n = len(x)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    fold_mses = []
    
    for i in range(n_folds):
        if _time_left() < 2:
            return np.inf
            
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            if model_type == 'spline':
                # Spline requires sorted data
                sort_idx = np.argsort(x_tr)
                x_tr_sorted = x_tr[sort_idx]
                y_tr_sorted = y_tr[sort_idx]
                
                # Check for duplicates
                if np.any(np.diff(x_tr_sorted) == 0):
                    unique_x, unique_indices = np.unique(x_tr_sorted, return_index=True)
                    if len(unique_x) < 4:
                        fold_mses.append(np.inf)
                        continue
                    y_tr_sorted = y_tr_sorted[unique_indices]
                    x_tr_sorted = unique_x
                    
                cs = CubicSpline(x_tr_sorted, y_tr_sorted)
                pred = cs(x_val)
            elif model_type == 'poly':
                deg = param
                X_tr_poly = _make_poly_matrix(x_tr, deg)
                X_val_poly = _make_poly_matrix(x_val, deg)
                
                # Ridge Regression (Tikhonov)
                # (X^T X + alpha I) w = X^T y
                alpha = 1.0
                A = X_tr_poly.T @ X_tr_poly + alpha * np.eye(deg + 1)
                b = X_tr_poly.T @ y_tr
                
                # Solve for w
                w = np.linalg.solve(A, b)
                pred = X_val_poly @ w
            else:
                return np.inf
                
            mse = np.mean((pred - y_val)**2)
            fold_mses.append(mse)
        except Exception:
            fold_mses.append(np.inf)
            
    if not fold_mses:
        return np.inf
    return np.mean(fold_mses)

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Sort data for Spline compatibility and general stability
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle constant y
    if np.std(y) < 1e-10:
        def predict(X):
            return np.full(X.shape[0], np.mean(y))
        return predict

    # Candidate models
    # 1. Spline (1 model)
    # 2. Polynomials (deg 1 to 8)
    
    best_mse = np.inf
    best_model = None
    best_param = None
    
    # Test Spline
    if _time_left() > 5:
        mse_spline = _cv_score(x_sorted, y_sorted, 'spline', None)
        if mse_spline < best_mse:
            best_mse = mse_spline
            best_model = 'spline'
            best_param = None

    # Test Polynomials
    for deg in range(1, 9):
        if _time_left() < 5:
            break
        mse_poly = _cv_score(x_sorted, y_sorted, 'poly', deg)
        if mse_poly < best_mse:
            best_mse = mse_poly
            best_model = 'poly'
            best_param = deg

    # If CV failed for all (unlikely), fallback to linear
    if best_model is None:
        best_model = 'poly'
        best_param = 1

    # Refit on all data
    if best_model == 'spline':
        # Ensure unique x for spline
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        if len(unique_x) < 4:
             # Fallback to linear if too few points
             best_model = 'poly'
             best_param = 1
        else:
             y_unique = y_sorted[unique_indices]
             try:
                 cs = CubicSpline(unique_x, y_unique)
                 def predict(X):
                     X_test = X[:, 0]
                     pred = cs(X_test)
                     return np.asarray(pred, dtype=float)
             except Exception:
                 # Fallback to linear
                 best_model = 'poly'
                 best_param = 1
                 
    if best_model == 'poly':
        deg = best_param
        X_poly = _make_poly_matrix(x_sorted, deg)
        
        # Ridge Regression (Tikhonov)
        alpha = 1.0
        A = X_poly.T @ X_poly + alpha * np.eye(deg + 1)
        b = X_poly.T @ y_sorted
        
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            w = np.linalg.lstsq(A, b, rcond=None)[0]
            
        def predict(X):
            X_test = X[:, 0].reshape(-1, 1)
            X_test_poly = _make_poly_matrix(X_test, deg)
            pred = X_test_poly @ w
            return np.asarray(pred, dtype=float)

    # Safety wrapper
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.asarray(res, dtype=float)
            # Replace non-finite with mean
            mean_y = np.mean(y)
            res = np.where(np.isfinite(res), res, mean_y)
            return res
        except Exception:
            return np.full(X.shape[0], np.mean(y))
            
    return safe_predict
# EVOLVE-BLOCK-END
