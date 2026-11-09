import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, differential_evolution
import inspect

def fit(X_train, y_train):
    """
    Fit a regression model. Primary strategy: Cubic spline interpolation for noiseless data.
    Secondary: RBF network with Gaussian kernels.
    Tertiary: Parametric curve fitting.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    # Handle potential NaNs or infinities in input
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
    
    # Remove duplicates for interpolation (keep first occurrence or average? Average is safer for noise, but task says noiseless. 
    # If duplicates exist with different y, it's not a function. If same y, just remove. 
    # Let's just keep unique x and corresponding y. If multiple y for same x, we take the first one for now, 
    # but since it's noiseless, they should be same.
    # Actually, np.unique returns sorted unique values. We need to map back to y.
    # Let's do it properly:
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    best_model = None
    best_rss = np.inf
    best_score = np.inf # Normalized MSE proxy

    var_y = np.var(y)
    if var_y == 0:
        var_y = 1e-10

    # 1. Try Cubic Spline Interpolation
    # This is extremely strong for smooth, noiseless 1D data.
    try:
        spline = CubicSpline(x_unique, y_unique)
        pred_spline = spline(x_unique)
        rss_spline = np.sum((y_unique - pred_spline)**2)
        
        norm_mse_spline = rss_spline / (len(y_unique) * var_y)
        
        if norm_mse_spline < best_score:
            best_score = norm_mse_spline
            best_model = 'spline'
            best_model_params = {'spline': spline}
            
        # If spline fits perfectly (or near perfectly), we can stop early
        if best_score < 1e-8:
            # Construct predictor immediately
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

    # 2. Try RBF Network (Global Radial Basis Function)
    # Use a subset of training points as centers to keep it efficient
    # Solve for weights via least squares
    try:
        # Use all unique points as centers if count is reasonable, else subsample
        n_centers = len(x_unique)
        if n_centers > 100:
            # Subsample centers
            indices = np.linspace(0, n_centers - 1, 100).astype(int)
            centers = x_unique[indices]
            y_centers = y_unique[indices]
        else:
            centers = x_unique
            y_centers = y_unique
            
        # Gaussian kernel width selection
        # Heuristic: distance between adjacent centers
        if len(centers) > 1:
            diffs = np.diff(centers)
            eps = np.mean(diffs)
        else:
            eps = 1.0
            
        # Try a few different widths
        best_rbf_score = np.inf
        best_rbf_weights = None
        best_rbf_eps = None
        
        for eps_factor in [0.5, 1.0, 2.0, 5.0]:
            eps_try = eps * eps_factor
            if eps_try < 1e-6:
                continue
                
            # Build kernel matrix K where K[i,j] = exp(-eps * (x_i - c_j)^2)
            # x_i are the training points, c_j are the centers
            K = np.exp(-eps_try * (x_unique[:, np.newaxis] - centers[np.newaxis, :])**2)
            
            # Solve K w = y
            try:
                weights, *_ = np.linalg.lstsq(K, y_unique, rcond=None)
                pred_rbf = K @ weights
                rss_rbf = np.sum((y_unique - pred_rbf)**2)
                norm_mse_rbf = rss_rbf / (len(y_unique) * var_y)
                
                if norm_mse_rbf < best_rbf_score:
                    best_rbf_score = norm_mse_rbf
                    best_rbf_weights = weights
                    best_rbf_eps = eps_try
                    
            except Exception:
                continue
        
        if best_rbf_score < best_score:
            best_score = best_rbf_score
            best_model = 'rbf'
            best_model_params = {
                'centers': centers,
                'weights': best_rbf_weights,
                'eps': best_rbf_eps
            }
            
    except Exception:
        pass

    # 3. Try Parametric Curve Fitting
    # This is a fallback if splines/RBFs fail (e.g. if data is very noisy or not smooth, though task says noiseless)
    nonlinear_forms = [
        (lambda x, a, b, c: a * np.sin(b * x) + c, "a*sin(b*x)+c"),
        (lambda x, a, b, c: a * np.cos(b * x) + c, "a*cos(b*x)+c"),
        (lambda x, a, b, c: a * np.exp(b * x) + c, "a*exp(b*x)+c"),
        (lambda x, a, b, c: a * np.exp(-b * x) + c, "a*exp(-b*x)+c"),
        (lambda x, a, b, c: a * np.exp(-b * x**2) + c, "a*exp(-b*x^2)+c"),
        (lambda x, a, b, c: a / (b + x) + c, "a/(b+x)+c"),
        (lambda x, a, b, c: a * np.log(b + x) + c, "a*log(b+x)+c"),
        (lambda x, a, b, c: a * np.tanh(b * x) + c, "a*tanh(b*x)+c"),
        (lambda x, a, b, c: a * x**b + c, "a*x^b+c"),
        (lambda x, a, b, c: a * x + b * x**2 + c, "quadratic"),
        (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, "cubic"),
    ]
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    x_range = np.max(x) - np.min(x)
    if x_range == 0:
        x_range = 1.0
        
    for func, name in nonlinear_forms:
        if remaining() < 5:
            break
            
        try:
            sig = inspect.signature(func)
            n_params = len(sig.parameters) - 1 # Exclude x
            
            # Generate initial guesses
            if n_params == 3:
                p0_list = [
                    [1, 1, 0],
                    [y_std, 1, y_mean],
                    [y_std, 2 * np.pi / x_range, y_mean],
                    [y_std, 1/x_range, y_mean],
                ]
            elif n_params == 4:
                p0_list = [
                    [1, 1, 1, 0],
                    [y_std, 0, 0, y_mean],
                    [y_std, 0, 0, y_mean],
                ]
            else:
                p0_list = [[1]*n_params]
                
            best_nl_rss = np.inf
            best_nl_params = None
            
            for p0 in p0_list:
                try:
                    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=2000)
                    pred = func(x, *popt)
                    rss = np.sum((y - pred)**2)
                    
                    if rss < best_nl_rss:
                        best_nl_rss = rss
                        best_nl_params = popt
                        
                except Exception:
                    continue
            
            if best_nl_rss < np.inf:
                norm_mse_nl = best_nl_rss / (n * var_y)
                if norm_mse_nl < best_score:
                    best_score = norm_mse_nl
                    best_model = 'nonlinear'
                    best_model_params = {'func': func, 'params': best_nl_params}
                    
        except Exception:
            continue

    # 4. Fallback: Mean
    if best_model is None:
        best_model = 'mean'
        best_model_params = {'mean': np.mean(y)}

    # 5. Construct Predictor
    
    if best_model == 'spline':
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

    elif best_model == 'rbf':
        centers = best_model_params['centers']
        weights = best_model_params['weights']
        eps = best_model_params['eps']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                K_pred = np.exp(-eps * (xi[:, np.newaxis] - centers[np.newaxis, :])**2)
                preds = K_pred @ weights
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))

    elif best_model == 'nonlinear':
        func = best_model_params['func']
        params = best_model_params['params']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = func(xi, *params)
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
