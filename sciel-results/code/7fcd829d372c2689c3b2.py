import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    d = X_train.shape[1]
    
    if d != 1:
        # Fallback for multi-dim, though task says 1D
        # Use simple linear regression or similar
        # For now, just return a constant predictor or linear
        # Let's stick to the 1D assumption as per task description
        # But to be safe:
        try:
             # Simple ridge regression
             X_aug = np.column_stack([X_train, np.ones(len(X_train))])
             w, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
             def predict(X):
                 X = np.asarray(X, dtype=np.float64)
                 if X.ndim == 1: X = X.reshape(-1, 1)
                 X_aug = np.column_stack([X, np.ones(len(X))])
                 return X_aug @ w
             return predict
        except:
             def predict(X):
                 return np.zeros(len(X))
             return predict

    # 1D Case
    x = X_train[:, 0]
    y = y_train
    
    # Sort
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_avg = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(y_avg, inverse, y_sorted)
        np.add.at(counts, inverse, 1)
        y_sorted = y_avg / counts
        x_sorted = unique_x
    
    # If only 1 or 2 points, spline won't work well or at all
    if len(x_sorted) < 4:
        # Use linear interpolation or constant
        if len(x_sorted) == 1:
            val = y_sorted[0]
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1: X = X.reshape(-1, 1)
                return np.full(X.shape[0], val)
            return predict
        
        # Linear interpolation for 2 or 3 points
        # np.interp handles this
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1: X = X.reshape(-1, 1)
            return np.interp(X[:, 0], x_sorted, y_sorted)
        return predict

    try:
        # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
        # This preserves monotonicity and avoids oscillations
        pchip = PchipInterpolator(x_sorted, y_sorted)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_pred = X[:, 0]
            return pchip(x_pred)
            
        return predict
    except Exception:
        # Fallback to Linear if PCHIP fails (e.g. numerical issues)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_pred = X[:, 0]
            return np.interp(x_pred, x_sorted, y_sorted)
        return predict
