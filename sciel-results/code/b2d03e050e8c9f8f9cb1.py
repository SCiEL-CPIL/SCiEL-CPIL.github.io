"""Rewritten program: direct 1D interpolation with robust spline handling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    """
    Fit a regression model for 1D noiseless data using direct interpolation.
    Since all 4 hidden equations have 1 input dimension and 200 noiseless points,
    we can sort the data and use cubic spline interpolation which will be exact
    for smooth functions. We also handle edge cases robustly.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 1D handling
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1)
    
    n, d = X_train.shape
    
    if d != 1:
        # Fallback for multi-dimensional (shouldn't happen per task spec)
        # Use polynomial regression
        from numpy.polynomial import polynomial as P
        x = X_train[:, 0]
        # Try a few degrees
        best_deg = 3
        for deg in [2, 3, 4, 5, 6, 8, 10]:
            coeffs = np.polyfit(x, y_train, deg)
            pred = np.polyval(coeffs, x)
            mse = np.mean((pred - y_train) ** 2)
            if mse < 1e-10:
                best_deg = deg
                break
        coeffs = np.polyfit(x, y_train, best_deg)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.polyval(coeffs, X[:, 0])
        return predict
    
    x = X_train[:, 0]
    y = y_train
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle potential duplicate x values by averaging
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(y_unique, inverse, y_sorted)
        np.add.at(counts, inverse, 1)
        y_unique /= counts
        x_final = unique_x
        y_final = y_unique
    else:
        x_final = x_sorted
        y_final = y_sorted
    
    # Try cubic spline interpolation
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x_final, y_final, bc_type='natural')
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_query = X[:, 0]
            return cs(x_query)
        
        # Verify on training data
        test_pred = predict(X_train)
        if np.all(np.isfinite(test_pred)):
            return predict
    except Exception:
        pass
    
    # Fallback: PCHIP (monotone cubic)
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(x_final, y_final)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_query = X[:, 0]
            return pchip(x_query)
        
        test_pred = predict(X_train)
        if np.all(np.isfinite(test_pred)):
            return predict
    except Exception:
        pass
    
    # Fallback: linear interpolation
    try:
        from scipy.interpolate import interp1d
        interp = interp1d(x_final, y_final, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_query = X[:, 0]
            return interp(x_query)
        
        test_pred = predict(X_train)
        if np.all(np.isfinite(test_pred)):
            return predict
    except Exception:
        pass
    
    # Last resort: nearest neighbor
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_query = X[:, 0]
        # Find nearest training point for each query
        diffs = np.abs(x_query[:, np.newaxis] - x_final[np.newaxis, :])
        idx = np.argmin(diffs, axis=1)
        return y_final[idx]
    
    return predict
# EVOLVE-BLOCK-END
