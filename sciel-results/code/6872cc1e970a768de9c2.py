import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0 # Safety margin
    
    X_flat = X_train.flatten()
    y_flat = y_train.flatten()
    
    # Check for constant y
    if np.var(y_flat) < 1e-10:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(np.asarray(X).flatten(), mean_y).reshape(np.asarray(X).shape)

    # Sort by X for interpolation
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle potential duplicate X values by averaging y
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
        
    if len(X_sorted) < 2:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(np.asarray(X).flatten(), mean_y).reshape(np.asarray(X).shape)

    # Try PCHIP (more stable than CubicSpline for arbitrary data)
    try:
        # PCHIP requires strictly increasing x
        if np.any(np.diff(X_sorted) <= 0):
             # This shouldn't happen after unique, but just in case
             raise ValueError("Non-increasing X")
        
        interpolator = PchipInterpolator(X_sorted, y_sorted)
        
        # Test evaluation to ensure no NaNs/Infs
        test_x = X_sorted[::10]
        test_y = interpolator(test_x)
        if not np.all(np.isfinite(test_y)):
            raise ValueError("Non-finite values in PCHIP")
            
    except Exception:
        # Fallback to linear interpolation
        interpolator = None
        use_linear = True
    else:
        use_linear = False

    def predict(X):
        X_eval = np.asarray(X).flatten()
        if use_linear:
            y_pred = np.interp(X_eval, X_sorted, y_sorted)
        else:
            y_pred = interpolator(X_eval)
            
        # Ensure finite output
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_pred.reshape(np.asarray(X).shape)

    return predict
