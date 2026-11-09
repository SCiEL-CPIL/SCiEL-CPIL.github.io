"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def fit(X_train, y_train):
    """
    Fit a regression model. 
    Given the task description (1D inputs, noiseless data), we use direct interpolation.
    PCHIP is preferred for its monotonicity preservation and lack of overshoot,
    which is crucial for normalized MSE scoring if the function has sharp features.
    """
    n, d = X_train.shape
    
    # Check for 1D input
    if d == 1:
        x = X_train[:, 0]
        y = y_train
        
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates: average y for same x
        unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < len(x_sorted):
            y_sum = np.zeros_like(unique_x)
            np.add.at(y_sum, inverse, y_sorted)
            y_unique = y_sum / counts
        else:
            unique_x = x_sorted
            y_unique = y_sorted
        
        # Use PCHIP for robust interpolation
        try:
            interpolator = PchipInterpolator(unique_x, y_unique)
            
            def predict(X):
                x_pred = X[:, 0]
                return interpolator(x_pred)
            
            return predict
        except Exception:
            # Fallback to linear interpolation if PCHIP fails
            try:
                interpolator = np.interp
                def predict(X):
                    x_pred = X[:, 0]
                    return np.interp(x_pred, unique_x, y_unique)
                return predict
            except Exception:
                # Last resort: constant mean
                mean_y = np.mean(y)
                def predict(X):
                    return np.full(len(X), mean_y)
                return predict
    else:
        # Fallback for multi-dimensional inputs (though task says 1D)
        # Use a simple polynomial regression as a generic fallback
        from sklearn.linear_model import LinearRegression
        # Since sklearn might not be available, use numpy lstsq with polynomial features
        degree = 2
        def build_features(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                for p in range(1, degree + 1):
                    cols.append(X[:, j] ** p)
            if d > 1:
                for i in range(d):
                    for j in range(i + 1, d):
                        cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        features = build_features(X_train)
        try:
            w, *_ = np.linalg.lstsq(features, y_train, rcond=None)
            def predict(X):
                return build_features(X) @ w
            return predict
        except Exception:
            mean_y = np.mean(y_train)
            def predict(X):
                return np.full(len(X), mean_y)
            return predict
# EVOLVE-BLOCK-END
