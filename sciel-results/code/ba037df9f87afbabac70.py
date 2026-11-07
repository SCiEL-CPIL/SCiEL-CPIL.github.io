import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    x = X.flatten()
    y_flat = y.flatten()
    
    if np.var(y_flat) < 1e-10:
        mean_y = np.mean(y_flat)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle duplicates by averaging
    unique_x, inverse_idx, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) != len(x_sorted):
        y_sum = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse_idx, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
    
    if len(x_sorted) < 4:
        mean_y = np.mean(y_sorted)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_mean
    
    x_min, x_max = x_sorted[0], x_sorted[-1]
    if x_max == x_min:
        val = y_sorted[0]
        def predict_single(X):
            return np.full(np.asarray(X).shape, val)
        return predict_single
    
    # Normalize to [-1, 1] for Chebyshev stability
    x_mid = (x_max + x_min) / 2.0
    x_half = (x_max - x_min) / 2.0
    x_norm = (x_sorted - x_mid) / x_half
    
    # We'll try multiple approaches and pick the best one via cross-validation
    # Approach 1: PCHIP interpolation
    # Approach 2: Chebyshev polynomial with adaptive degree
    # Approach 3: Cubic spline with adaptive smoothing
    
    best_predict_fn = None
    best_val_err = float('inf')
    
    # Cross-validation split
    cv_split = int(len(x_sorted) * 0.8)
    x_tr, y_tr = x_sorted[:cv_split], y_sorted[:cv_split]
    x_val, y_val = x_sorted[cv_split:], y_sorted[cv_split:]
    
    x_tr_norm = (x_tr - x_mid) / x_half
    x_val_norm = (x_val - x_mid) / x_half
    
    # Approach 1: PCHIP
    try:
        pchip = PchipInterpolator(x_tr, y_tr)
        y_pred_val = pchip(x_val)
        val_err = np.mean((y_pred_val - y_val) ** 2)
        
        if val_err < best_val_err:
            best_val_err = val_err
            # Fit on all data
            pchip_full = PchipInterpolator(x_sorted, y_sorted)
            
            def predict_pchip(X):
                x_eval = np.asarray(X).flatten()
                y_pred = pchip_full(x_eval)
                return y_pred.reshape(np.asarray(X).shape)
            
            best_predict_fn = predict_pchip
    except Exception:
        pass
    
    # Approach 2: Chebyshev with adaptive degree
    try:
        best_cheb_c = None
        best_cheb_deg = 10
        min_cheb_err = float('inf')
        
        for deg in [10, 20, 30, 50, 80, 100, 150, 200]:
            if time.time() > deadline - 10:
                break
            try:
                c = np.polynomial.chebyshev.chebfit(x_tr_norm, y_tr, deg)
                y_pred = np.polynomial.chebyshev.chebval(x_val_norm, c)
                err = np.mean((y_pred - y_val) ** 2)
                if err < min_cheb_err:
                    min_cheb_err = err
                    best_cheb_c = c
                    best_cheb_deg = deg
            except Exception:
                continue
        
        if best_cheb_c is not None:
            if min_cheb_err < best_val_err:
                best_val_err = min_cheb_err
                # Fit on all data with best degree
                c_final = np.polynomial.chebyshev.chebfit(x_norm, y_sorted, best_cheb_deg)
                
                def predict_cheb(X):
                    x_eval = np.asarray(X).flatten()
                    x_eval_norm = (x_eval - x_mid) / x_half
                    y_pred = np.polynomial.chebyshev.chebval(x_eval_norm, c_final)
                    return y_pred.reshape(np.asarray(X).shape)
                
                best_predict_fn = predict_cheb
    except Exception:
        pass
    
    # Approach 3: Cubic spline with smoothing
    try:
        from scipy.interpolate import UnivariateSpline
        
        best_spl = None
        min_spl_err = float('inf')
        
        for s in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
            if time.time() > deadline - 5:
                break
            try:
                spl = UnivariateSpline(x_tr, y_tr, s=s)
                y_pred = spl(x_val)
                err = np.mean((y_pred - y_val) ** 2)
                if err < min_spl_err:
                    min_spl_err = err
                    best_spl = spl
            except Exception:
                continue
        
        if best_spl is not None and min_spl_err < best_val_err:
            best_val_err = min_spl_err
            spl_full = UnivariateSpline(x_sorted, y_sorted, s=best_spl.s)
            
            def predict_spline(X):
                x_eval = np.asarray(X).flatten()
                y_pred = spl_full(x_eval)
                return y_pred.reshape(np.asarray(X).shape)
            
            best_predict_fn = predict_spline
    except Exception:
        pass
    
    # If no approach worked, fall back to Chebyshev degree 10
    if best_predict_fn is None:
        try:
            c = np.polynomial.chebyshev.chebfit(x_norm, y_sorted, 10)
            def predict_fallback(X):
                x_eval = np.asarray(X).flatten()
                x_eval_norm = (x_eval - x_mid) / x_half
                y_pred = np.polynomial.chebyshev.chebval(x_eval_norm, c)
                return y_pred.reshape(np.asarray(X).shape)
            best_predict_fn = predict_fallback
        except Exception:
            mean_y = np.mean(y_sorted)
            def predict_final_fallback(X):
                return np.full(np.asarray(X).shape, mean_y)
            best_predict_fn = predict_final_fallback
    
    # Ensure predictions are finite
    def predict_safe(X):
        result = best_predict_fn(X)
        result = np.asarray(result, dtype=float)
        # Replace any non-finite values with the mean of training y
        if not np.all(np.isfinite(result)):
            result = np.where(np.isfinite(result), result, np.mean(y_sorted))
        return result
    
    return predict_safe
