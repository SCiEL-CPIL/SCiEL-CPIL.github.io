import numpy as np
import scipy.interpolate as interpolate
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    x = X_train.flatten()
    y = y_train.flatten()
    y_shape = y_train.shape
    
    # Sort
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_x, unique_indices, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    if len(unique_x) < len(x_sorted):
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, unique_indices, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
        
    n = len(x_sorted)
    if n < 4:
        # Not enough points for spline/pchip, use linear or constant
        mean_y = np.mean(y_sorted)
        def predict_const(X):
            return np.full(X.shape[0], mean_y).reshape(y_shape)
        return predict_const

    # Validation split
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    x_tr = x_sorted[train_idx]
    y_tr = y_sorted[train_idx]
    x_val = x_sorted[val_idx]
    y_val = y_sorted[val_idx]
    
    models = {}
    
    # 1. PCHIP
    try:
        pchip = interpolate.PchipInterpolator(x_tr, y_tr)
        y_pred = pchip(x_val)
        err = np.mean((y_pred - y_val)**2)
        if np.isfinite(err):
            models['pchip'] = (err, 'pchip')
    except Exception:
        pass
        
    # 2. Cubic Spline
    try:
        spline = interpolate.CubicSpline(x_tr, y_tr)
        y_pred = spline(x_val)
        err = np.mean((y_pred - y_val)**2)
        if np.isfinite(err):
            models['spline'] = (err, 'spline')
    except Exception:
        pass
        
    # 3. Polyfit degree 8
    try:
        coeffs = np.polyfit(x_tr, y_tr, 8)
        y_pred = np.polyval(coeffs, x_val)
        err = np.mean((y_pred - y_val)**2)
        if np.isfinite(err):
            models['poly8'] = (err, 'poly8')
    except Exception:
        pass
        
    # 4. Polyfit degree 12
    try:
        coeffs = np.polyfit(x_tr, y_tr, 12)
        y_pred = np.polyval(coeffs, x_val)
        err = np.mean((y_pred - y_val)**2)
        if np.isfinite(err):
            models['poly12'] = (err, 'poly12')
    except Exception:
        pass

    if not models:
        # Fallback to linear
        def predict_linear(X):
            x_test = X.flatten()
            res = np.interp(x_test, x_sorted, y_sorted)
            return res.reshape(y_shape)
        return predict_linear

    # Pick best
    best_err, best_type = min(models.items(), key=lambda item: item[0])
    
    # Refit on all data
    if best_type == 'pchip':
        final_model = interpolate.PchipInterpolator(x_sorted, y_sorted)
        def predict(X):
            x_test = X.flatten()
            res = final_model(x_test)
            return res.reshape(y_shape)
    elif best_type == 'spline':
        final_model = interpolate.CubicSpline(x_sorted, y_sorted)
        def predict(X):
            x_test = X.flatten()
            res = final_model(x_test)
            return res.reshape(y_shape)
    elif best_type == 'poly8':
        coeffs = np.polyfit(x_sorted, y_sorted, 8)
        def predict(X):
            x_test = X.flatten()
            res = np.polyval(coeffs, x_test)
            return res.reshape(y_shape)
    elif best_type == 'poly12':
        coeffs = np.polyfit(x_sorted, y_sorted, 12)
        def predict(X):
            x_test = X.flatten()
            res = np.polyval(coeffs, x_test)
            return res.reshape(y_shape)
            
    return predict
