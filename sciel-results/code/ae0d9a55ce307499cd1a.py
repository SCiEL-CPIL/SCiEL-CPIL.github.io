"""Symbolic regression via Piecewise Cubic Hermite Interpolation (PCHIP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    x = X[:, 0]
    n = len(x)
    
    # Sort data by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        x_sorted = unique_x
        y_sorted = y_avg
        n = len(x_sorted)
        
    # Create the interpolator once
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
        
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            x_pred = X_pred[:, 0]
            
            y_pred = pchip(x_pred)
            
            if not np.all(np.isfinite(y_pred)):
                y_pred = np.where(np.isfinite(y_pred), y_pred, np.mean(y_sorted))
                
            return y_pred
            
    except Exception as e:
        # Fallback to simple linear interpolation with linear extrapolation
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            x_pred = X_pred[:, 0]
            
            # Linear interpolation for interior, linear extrapolation for boundaries
            # np.interp does linear interpolation and clamps to endpoints, which is not ideal for extrapolation
            # We implement custom logic
            
            y_pred = np.zeros_like(x_pred)
            
            # Interior
            mask_interior = (x_pred >= x_sorted[0]) & (x_pred <= x_sorted[-1])
            if np.any(mask_interior):
                y_pred[mask_interior] = np.interp(x_pred[mask_interior], x_sorted, y_sorted)
                
            # Left extrapolation
            mask_left = x_pred < x_sorted[0]
            if np.any(mask_left):
                if n >= 2:
                    slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
                    y_pred[mask_left] = y_sorted[0] + slope * (x_pred[mask_left] - x_sorted[0])
                else:
                    y_pred[mask_left] = y_sorted[0]
                    
            # Right extrapolation
            mask_right = x_pred > x_sorted[-1]
            if np.any(mask_right):
                if n >= 2:
                    slope = (y_sorted[-1] - y_sorted[-2]) / (x_sorted[-1] - x_sorted[-2])
                    y_pred[mask_right] = y_sorted[-1] + slope * (x_pred[mask_right] - x_sorted[-1])
                else:
                    y_pred[mask_right] = y_sorted[-1]
                    
            return y_pred

    return predict

# EVOLVE-BLOCK-END
