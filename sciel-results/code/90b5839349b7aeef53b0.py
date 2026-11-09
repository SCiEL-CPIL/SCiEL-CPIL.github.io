import time
import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator
import warnings

def fit(X_train, y_train):
    """
    Fit a regression model.
    Strategy:
    1. Try RBF Interpolation with thin_plate_spline (robust for 1D smooth data).
    2. Try Cubic Spline Interpolation.
    3. Fallback to Mean.
    """
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    # Handle potential NaNs or infinities
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    def remaining():
        return deadline - time.time()
    
    # Sort by x for interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates for interpolation
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]
    
    x2d_unique = x_unique.reshape(-1, 1)
    
    best_model = None
    best_score = np.inf
    best_model_params = None

    var_y = np.var(y)
    if var_y < 1e-12:
        var_y = 1e-10

    # 1. Try RBF Interpolation (Thin Plate Spline)
    # This is often more stable than Cubic Spline for arbitrary 1D data
    try:
        if len(x_unique) > 2:
            # Use a small epsilon for regularization if needed, but thin_plate_spline is usually fine
            # RBFInterpolator can be slow for large N, but N=200 is fine.
            rbf_interp = RBFInterpolator(x2d_unique, y_unique, kernel='thin_plate_spline', epsilon=1e-8)
            
            # Evaluate on training points to check fit
            pred_rbf = rbf_interp(x2d_unique)
            rss_rbf = np.sum((y_unique - pred_rbf)**2)
            norm_mse_rbf = rss_rbf / (len(y_unique) * var_y)
            
            if norm_mse_rbf < best_score:
                best_score = norm_mse_rbf
                best_model = 'rbf'
                best_model_params = {'interp': rbf_interp}
                
            # If RBF fits perfectly, we can stop early
            if best_score < 1e-8:
                def predict(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0].reshape(-1, 1)
                    try:
                        preds = best_model_params['interp'](xi)
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, np.median(y))
                        return preds
                    except Exception:
                        return np.full(len(X_in), np.median(y))
                return predict
    except Exception:
        pass

    # 2. Try Cubic Spline Interpolation
    try:
        if len(x_unique) > 1:
            spline = CubicSpline(x_unique, y_unique)
            pred_spline = spline(x_unique)
            rss_spline = np.sum((y_unique - pred_spline)**2)
            
            norm_mse_spline = rss_spline / (len(y_unique) * var_y)
            
            if norm_mse_spline < best_score:
                best_score = norm_mse_spline
                best_model = 'spline'
                best_model_params = {'spline': spline}
                
            # If spline fits perfectly, we can stop early
            if best_score < 1e-8:
                spl = best_model_params['spline']
                def predict(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0]
                    try:
                        preds = spl(xi)
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, np.median(y))
                        return preds
                    except Exception:
                        return np.full(len(X_in), np.median(y))
                return predict
    except Exception:
        pass

    # 3. Fallback: Mean
    if best_model is None:
        best_model = 'mean'
        best_model_params = {'mean': np.mean(y)}

    # 4. Construct Predictor
    
    if best_model == 'rbf':
        interp = best_model_params['interp']
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0].reshape(-1, 1)
            try:
                preds = interp(xi)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))

    elif best_model == 'spline':
        spline = best_model_params['spline']
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = spline(xi)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))
                
    else:
        mean_y = best_model_params['mean']
        def predict(X_in):
            return np.full(len(X_in), mean_y)

    return predict
