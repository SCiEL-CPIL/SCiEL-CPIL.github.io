import numpy as np
from scipy.interpolate import CubicSpline
import time

def fit(X_train, y_train):
    """
    Fits a cubic spline interpolant to the data.
    Since the data is noiseless and dense (200 points), interpolation
    should yield near-zero error on test points within the training range.
    """
    X = np.asarray(X_train).flatten()
    y = np.asarray(y_train).flatten()
    
    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle potential duplicates in X by averaging Y
    # CubicSpline requires strictly increasing x
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) < len(X_sorted):
        y_sum = np.zeros(len(unique_X))
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
    
    try:
        # s=0 for interpolation
        cs = CubicSpline(X_sorted, y_sorted)
    except Exception:
        # Fallback to polynomial if spline fails (e.g., numerical issues)
        # Use degree 10 polynomial
        try:
            coeffs = np.polyfit(X_sorted, y_sorted, 10)
            def predict_poly(X):
                X = np.asarray(X).flatten()
                return np.polyval(coeffs, X)
            return predict_poly
        except Exception:
            # Last resort: mean
            mean_y = np.mean(y)
            def predict_mean(X):
                X = np.asarray(X).flatten()
                return np.full_like(X, mean_y, dtype=float)
            return predict_mean

    def predict(X):
        X = np.asarray(X).flatten()
        return cs(X)

    return predict
