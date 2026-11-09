# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
import warnings

def fit(X_train, y_train):
    """
    Fit a regression model using cubic spline interpolation.
    Strategy:
    1. Sort by input feature.
    2. Handle duplicate x-values by averaging y-values.
    3. Use scipy.interpolate.CubicSpline for stable, exact interpolation.
       CubicSpline uses a banded matrix solver for the tridiagonal system,
       which is numerically stable for 200 points.
    4. Fallback to PchipInterpolator if CubicSpline fails.
    5. Fallback to linear interpolation as a last resort.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 1D input handling
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if n_features > 1:
        # Fallback for multi-feature (should not happen per task description)
        # Use polynomial regression
        X_1d = X[:, 0]
        degree = 5
        powers = np.vander(X_1d, N=degree+1, increasing=True)
        try:
            w, *_ = np.linalg.lstsq(powers, y, rcond=None)
        except np.linalg.LinAlgError:
            w = np.zeros(degree+1)
        
        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            X_1d = X_test[:, 0]
            powers_test = np.vander(X_1d, N=degree+1, increasing=True)
            return powers_test @ w
        return predict_poly

    # 1D case
    x = X[:, 0]
    y = y.reshape(-1)
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicate x values: average y values
    unique_x, unique_indices, inverse_indices = np.unique(x_sorted, return_index=True, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_sum = np.zeros(len(unique_x))
        y_count = np.zeros(len(unique_x))
        np.add.at(y_sum, inverse_indices, y_sorted)
        np.add.at(y_count, inverse_indices, 1)
        y_unique = y_sum / y_count
        x_sorted = unique_x
        y_sorted = y_unique
    
    # Ensure finite values
    mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    if not np.all(mask):
        x_sorted = x_sorted[mask]
        y_sorted = y_sorted[mask]
        
    if len(x_sorted) < 4:
        interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        def predict_linear(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return interp_func(x_test)
        return predict_linear

    # Try CubicSpline
    try:
        spline = CubicSpline(x_sorted, y_sorted)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return spline(x_test)
        return predict
    except Exception:
        pass

    # Try PchipInterpolator
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return pchip(x_test)
        return predict
    except Exception:
        pass

    # Fallback to linear interpolation
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    def predict_linear(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        return interp_func(x_test)
    return predict_linear
# EVOLVE-BLOCK-END
