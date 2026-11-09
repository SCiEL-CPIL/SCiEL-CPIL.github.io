import time
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
import inspect
import itertools
import operator
from functools import reduce

def _safe_eval(expr_func, x, params):
    """Safely evaluate a parametric function, returning inf on failure."""
    try:
        result = expr_func(x, *params)
        result = np.asarray(result, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            return np.inf, result
        return np.sum((y_global - result)**2), result
    except Exception:
        return np.inf, None

# Global for use in safe_eval
y_global = None

def _build_symbolic_library():
    """Build a library of candidate symbolic expressions with their parameter counts."""
    # Each entry: (name, func, n_params)
    # We'll use a rich set of forms
    lib = []
    
    # Polynomials up to degree 4
    lib.append(("poly1", lambda x, a, b: a*x + b, 2))
    lib.append(("poly2", lambda x, a, b, c: a*x**2 + b*x + c, 3))
    lib.append(("poly3", lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, 4))
    lib.append(("poly4", lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, 5))
    
    # Trig
    lib.append(("sin1", lambda x, a, b, c: a*np.sin(b*x) + c, 3))
    lib.append(("cos1", lambda x, a, b, c: a*np.cos(b*x) + c, 3))
    lib.append(("sin2", lambda x, a, b, c, d: a*np.sin(b*x + c) + d, 4))
    lib.append(("cos2", lambda x, a, b, c, d: a*np.cos(b*x + c) + d, 4))
    lib.append(("sincos", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*np.cos(d*x) + e, 5))
    
    # Exp
    lib.append(("exp1", lambda x, a, b, c: a*np.exp(b*x) + c, 3))
    lib.append(("expneg", lambda x, a, b, c: a*np.exp(-b*x) + c, 3))
    lib.append(("gauss", lambda x, a, b, c, d: a*np.exp(-b*(x-c)**2) + d, 4))
    lib.append(("lorentz", lambda x, a, b, c: a / (b + (x-c)**2) + d if False else a/(b+(x-c)**2)+d, 4))
    
    # Log
    lib.append(("log1", lambda x, a, b, c: a*np.log(b + x) + c, 3))
    lib.append(("logabs", lambda x, a, b, c: a*np.log(np.abs(x) + b) + c, 3))
    
    # Power
    lib.append(("pow1", lambda x, a, b, c: a * np.sign(x) * np.abs(x)**b + c, 3))
    lib.append(("pow2", lambda x, a, b, c, d: a * np.sign(x) * np.abs(x)**b + c*x + d, 4))
    
    # Rational
    lib.append(("rat1", lambda x, a, b, c: a*x/(b + x) + c, 3))
    lib.append(("rat2", lambda x, a, b, c, d: a*x/(b + c*x) + d, 4))
    lib.append(("rat3", lambda x, a, b, c: a / (b + x) + c, 3))
    lib.append(("rat4", lambda x, a, b, c, d: a / (b + c*x) + d, 4))
    
    # Tanh
    lib.append(("tanh1", lambda x, a, b, c: a*np.tanh(b*x) + c, 3))
    lib.append(("tanh2", lambda x, a, b, c, d: a*np.tanh(b*x + c) + d, 4))
    
    # Sigmoid
    lib.append(("sig1", lambda x, a, b, c: a / (1 + np.exp(-b*x)) + c, 3))
    
    # Piecewise-ish via abs
    lib.append(("abs1", lambda x, a, b, c: a*np.abs(x) + b*x + c, 3))
    lib.append(("abs2", lambda x, a, b, c, d: a*np.abs(x - b) + c*x + d, 4))
    
    # Composite: trig + poly
    lib.append(("sinpoly", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*x**2 + d*x + e, 5))
    lib.append(("exppoly", lambda x, a, b, c, d: a*np.exp(-b*x**2) + c*x + d, 4))
    
    return lib

def _initial_params_for_form(func, n_params, x, y):
    """Generate multiple initial parameter guesses for a given form."""
    y_mean = np.mean(y)
    y_std = np.std(y) if np.std(y) > 1e-12 else 1.0
    x_range = np.max(x) - np.min(x)
    if x_range < 1e-12:
        x_range = 1.0
    x_mean = np.mean(x)
    
    guesses = []
    
    # Generic: small random values
    for _ in range(3):
        p = np.random.uniform(-y_std, y_std, size=n_params)
        p[-1] = y_mean  # last param often offset
        guesses.append(p)
    
    # Specific heuristics
    if n_params == 2:
        guesses.append([y_std / x_range, y_mean])
    elif n_params == 3:
        guesses.append([y_std, 1.0, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, y_mean])
        guesses.append([y_std, 1.0/x_range, y_mean])
        guesses.append([y_std, 0.5, y_mean])
    elif n_params == 4:
        guesses.append([y_std, 1.0, 0.0, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, 0.0, y_mean])
        guesses.append([y_std, 1.0, x_mean, y_mean])
        guesses.append([y_std, 1.0/x_range, 0.0, y_mean])
    elif n_params == 5:
        guesses.append([y_std, 1.0, 0.0, y_std, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, 0.0, y_std/x_range, y_mean])
        guesses.append([y_std, 1.0, 0.0, 0.0, y_mean])
    
    return guesses

def fit(X_train, y_train):
    global y_global
    
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    y_global = y
    
    var_y = np.var(y)
    if var_y < 1e-12:
        var_y = 1e-10
    
    def remaining():
        return deadline - time.time()
    
    # Build symbolic library
    sym_lib = _build_symbolic_library()
    
    best_rss = np.inf
    best_func = None
    best_params = None
    best_name = "none"
    
    # Sort x for potential interpolation fallback
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Strategy 1: Try each symbolic form with multiple initial guesses
    for name, func, n_params in sym_lib:
        if remaining() < 3:
            break
        
        try:
            # Determine bounds for differential_evolution if needed
            y_scale = np.std(y) if np.std(y) > 1e-12 else 1.0
            x_scale = np.max(x) - np.min(x)
            if x_scale < 1e-12:
                x_scale = 1.0
            
            # Bounds: param 0 often amplitude, param 1 often frequency, last often offset
            bounds = []
            for i in range(n_params):
                if i == 0:
                    bounds.append((-5*y_scale, 5*y_scale))
                elif i == 1 and n_params >= 3:
                    # Frequency-like
                    bounds.append((0.01, 10.0/x_scale))
                elif i == n_params - 1:
                    # Offset
                    bounds.append((-5*y_scale, 5*y_scale))
                else:
                    bounds.append((-5*y_scale, 5*y_scale))
            
            # First try curve_fit with multiple initial guesses (fast)
            p0_list = _initial_params_for_form(func, n_params, x, y)
            
            best_local_rss = np.inf
            best_local_params = None
            
            for p0 in p0_list:
                if remaining() < 1:
                    break
                try:
                    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                    pred = func(x, *popt)
                    pred = np.asarray(pred, dtype=np.float64)
                    if not np.all(np.isfinite(pred)):
                        continue
                    rss = np.sum((y - pred)**2)
                    if rss < best_local_rss:
                        best_local_rss = rss
                        best_local_params = popt
                except Exception:
                    continue
            
            if best_local_rss < np.inf and best_local_rss < best_rss:
                best_rss = best_local_rss
                best_func = func
                best_params = best_local_params
                best_name = name
            
            # If curve_fit didn't get a great fit, try differential_evolution for harder forms
            if best_local_rss > 1e-6 * n * var_y and n_params <= 5:
                if remaining() < 5:
                    break
                try:
                    def objective(params):
                        try:
                            pred = func(x, *params)
                            pred = np.asarray(pred, dtype=np.float64)
                            if not np.all(np.isfinite(pred)):
                                return 1e15
                            return np.sum((y - pred)**2)
                        except Exception:
                            return 1e15
                    
                    de_result = differential_evolution(
                        objective, bounds, maxiter=100, popsize=20,
                        seed=42, tol=1e-10, polish=True
                    )
                    if de_result.fun < best_local_rss:
                        best_local_rss = de_result.fun
                        best_local_params = de_result.x
                        if best_local_rss < best_rss:
                            best_rss = best_local_rss
                            best_func = func
                            best_params = best_local_params
                            best_name = name + "_de"
                except Exception:
                    pass
                    
        except Exception:
            continue
    
    # Strategy 2: Cubic spline as fallback/comparison
    try:
        from scipy.interpolate import CubicSpline
        x_unique, unique_indices = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_indices]
        if len(x_unique) >= 4:
            spline = CubicSpline(x_unique, y_unique)
            pred_spline = spline(x_unique)
            rss_spline = np.sum((y_unique - pred_spline)**2)
            if rss_spline < best_rss:
                best_rss = rss_spline
                best_func = None  # Use spline
                best_params = None
                best_name = "spline"
    except Exception:
        pass
    
    # Strategy 3: RBF as another fallback
    if best_name != "spline":
        try:
            x_unique, unique_indices = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_indices]
            n_centers = len(x_unique)
            if n_centers > 80:
                indices = np.linspace(0, n_centers - 1, 80).astype(int)
                centers = x_unique[indices]
                y_centers = y_unique[indices]
            else:
                centers = x_unique
                y_centers = y_unique
            
            if len(centers) > 1:
                diffs = np.diff(centers)
                eps_base = np.mean(diffs)
            else:
                eps_base = 1.0
            
            best_rbf_rss = np.inf
            best_rbf_weights = None
            best_rbf_eps = None
            
            for eps_factor in [0.5, 1.0, 2.0, 5.0]:
                eps_try = eps_base * eps_factor
                if eps_try < 1e-8:
                    continue
                K = np.exp(-eps_try * (x_unique[:, np.newaxis] - centers[np.newaxis, :])**2)
                try:
                    weights, *_ = np.linalg.lstsq(K, y_unique, rcond=None)
                    pred_rbf = K @ weights
                    rss_rbf = np.sum((y_unique - pred_rbf)**2)
                    if rss_rbf < best_rbf_rss:
                        best_rbf_rss = rss_rbf
                        best_rbf_weights = weights
                        best_rbf_eps = eps_try
                except Exception:
                    continue
            
            if best_rbf_rss < best_rss and best_rbf_weights is not None:
                best_rss = best_rbf_rss
                best_func = None
                best_params = None
                best_name = "rbf"
                # Store RBF params
                best_rbf_data = (centers, best_rbf_weights, best_rbf_eps)
            else:
                best_rbf_data = None
        except Exception:
            best_rbf_data = None
    else:
        best_rbf_data = None
    
    # Construct predictor
    median_y = np.median(y)
    
    if best_name == "spline":
        try:
            x_unique, unique_indices = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_indices]
            spline = CubicSpline(x_unique, y_unique)
            
            def predict(X_in):
                X_in = np.asarray(X_in, dtype=np.float64)
                xi = X_in[:, 0]
                try:
                    preds = spline(xi)
                    preds = np.asarray(preds, dtype=np.float64)
                    if not np.all(np.isfinite(preds)):
                        preds = np.where(np.isfinite(preds), preds, median_y)
                    return preds
                except Exception:
                    return np.full(len(X_in), median_y)
        except Exception:
            def predict(X_in):
                return np.full(len(np.asarray(X_in).shape[0]), median_y)
    
    elif best_name == "rbf" and best_rbf_data is not None:
        centers, weights, eps = best_rbf_data
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                K_pred = np.exp(-eps * (xi[:, np.newaxis] - centers[np.newaxis, :])**2)
                preds = K_pred @ weights
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, median_y)
                return preds
            except Exception:
                return np.full(len(X_in), median_y)
    
    elif best_func is not None and best_params is not None:
        func = best_func
        params = best_params
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = func(xi, *params)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, median_y)
                return preds
            except Exception:
                return np.full(len(X_in), median_y)
    
    else:
        def predict(X_in):
            return np.full(len(np.asarray(X_in).shape[0]), median_y)
    
    return predict
