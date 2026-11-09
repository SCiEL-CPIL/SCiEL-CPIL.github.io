# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize_scalar
import warnings


def fit(X_train, y_train):
    """
    Fit a regression model using piecewise polynomial interpolation with adaptive knot placement.
    Strategy:
    1. Sort by input feature.
    2. Use RANSAC-like robust fitting to identify inliers (though data is noiseless, this helps if there's any numerical noise).
    3. Fit a high-order spline (cubic or quartic) with adaptive smoothing parameter.
    4. If the function is non-monotonic or has complex structure, use a higher degree or more knots.
    5. For 1D input, we can use scipy's UnivariateSpline with optimal smoothing.
    
    Since the data is noiseless, we can use interpolation (smoothing=0) or very low smoothing.
    However, to avoid overfitting to potential numerical noise and to handle any non-uniform spacing well,
    we'll use a robust approach:
    - First, try a cubic spline interpolation.
    - If the input is 1D, we can also try to detect if the function is periodic or has other properties.
    
    Given the time limit and the fact that there are 4 equations, we need to be efficient.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 1D input handling
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # If multiple features, we'd need a different approach, but the task specifies 1 input for each equation.
    # So we assume n_features == 1.
    if n_features > 1:
        # Fallback to polynomial regression for multi-feature case
        # But the task says 1 input, so this shouldn't happen.
        from numpy.polynomial import polynomial as P
        # Use degree 4 polynomial
        X_1d = X[:, 0]
        powers = np.column_stack([X_1d**i for i in range(5)])
        w, *_ = np.linalg.lstsq(powers, y, rcond=None)
        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            X_1d = X_test[:, 0]
            powers_test = np.column_stack([X_1d**i for i in range(5)])
            return powers_test @ w
        return predict_poly
    
    # 1D case
    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        # Average y values for duplicate x
        y_unique = np.array([y_sorted[unique_indices[i]:unique_indices[i]+1].mean() if i == len(unique_indices)-1 
                            else y_sorted[unique_indices[i]:unique_indices[i+1]].mean() 
                            for i in range(len(unique_indices))])
        # This is a bit tricky, let's do it properly
        y_unique = np.zeros(len(unique_x))
        for i, idx in enumerate(unique_indices):
            if i < len(unique_indices) - 1:
                next_idx = unique_indices[i+1]
                y_unique[i] = y_sorted[idx:next_idx].mean()
            else:
                y_unique[i] = y_sorted[idx:].mean()
        x_sorted = unique_x
        y_sorted = y_unique
    
    # Use UnivariateSpline with k=3 (cubic) and s=0 (interpolation)
    # Since data is noiseless, s=0 should give exact interpolation
    try:
        spline = UnivariateSpline(x_sorted, y_sorted, k=3, s=0)
    except Exception:
        # Fallback to linear interpolation
        interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        def predict_linear(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return interp_func(x_test)
        return predict_linear
    
    # Check if the spline is well-behaved
    # For noiseless data, s=0 interpolation should be exact
    # But we might want to add a small amount of smoothing if the function is noisy
    # Since the task says noiseless, we'll stick with s=0
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        return spline(x_test)
    
    return predict
# EVOLVE-BLOCK-END
