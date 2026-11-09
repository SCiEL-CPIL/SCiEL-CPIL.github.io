"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    X = X_train[:, 0]
    y = y_train
    
    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates by keeping the first occurrence (or averaging, but noiseless implies same value)
    X_unique, unique_indices = np.unique(X_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]
    
    # Fit Cubic Spline
    try:
        spline = CubicSpline(X_unique, y_unique, bc_type='not-a-knot')
    except Exception:
        # Fallback to linear interpolation if spline fails
        def predict(X_new):
            x_new = X_new[:, 0]
            return np.interp(x_new, X_unique, y_unique)
        return predict

    def predict(X_new):
        x_new = X_new[:, 0]
        return spline(x_new)

    return predict

# EVOLVE-BLOCK-END
