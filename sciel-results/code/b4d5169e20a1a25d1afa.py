import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
import itertools

def fit(X_train, y_train):
    """
    Fit a regression model. Primary strategy: Broad symbolic regression search
    over a library of candidate functions. Fallback: PCHIP interpolation.
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

    # Sort by x for interpolation fallbacks
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates for interpolation
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    best_model = None
    best_rss = np.inf
    best_score = np.inf # Normalized MSE proxy

    # 1. Try Linear Basis Combination (Polynomials + Trig + Exp + Log + etc.)
    # This covers: a0 + a1*x + ... + ak*fk(x)
    basis_funcs = []
    basis_names = []
    
    # Polynomials
    for p in range(1, 11):
        basis_funcs.append(lambda x, p=p: x**p)
        basis_names.append(f"x^{p}")
        
    # Trig
    for k in [1, 2, 3, 4, 5]:
        basis_funcs.append(lambda x, k=k: np.sin(k * x))
        basis_names.append(f"sin({k}x)")
        basis_funcs.append(lambda x, k=k: np.cos(k * x))
        basis_names.append(f"cos({k}x)")
        
    # Exponential / Log / Hyperbolic
    basis_funcs.append(lambda x: np.exp(x))
    basis_names.append("exp(x)")
    basis_funcs.append(lambda x: np.exp(-x))
    basis_names.append("exp(-x)")
    basis_funcs.append(lambda x: np.exp(-x**2))
    basis_names.append("exp(-x^2)")
    basis_funcs.append(lambda x: np.exp(x**2))
    basis_names.append("exp(x^2)")
    
    basis_funcs.append(lambda x: np.tanh(x))
    basis_names.append("tanh(x)")
    basis_funcs.append(lambda x: np.sinh(x))
    basis_names.append("sinh(x)")
    basis_funcs.append(lambda x: np.cosh(x))
    basis_names.append("cosh(x)")
    
    # Safe Log and Inverse
    basis_funcs.append(lambda x: np.log(np.abs(x) + 1e-9))
    basis_names.append("log|x|")
    basis_funcs.append(lambda x: np.log(x + 10)) # For x >= -10
    basis_names.append("log(x+10)")
    basis_funcs.append(lambda x: 1.0 / (x + 1e-9))
    basis_names.append("1/x")
    basis_funcs.append(lambda x: 1.0 / (1.0 + x**2))
    basis_names.append("1/(1+x^2)")
    basis_funcs.append(lambda x: 1.0 / (1.0 + np.exp(-x))) # Sigmoid
    basis_names.append("sigmoid(x)")
    
    # Mixed
    basis_funcs.append(lambda x: x * np.sin(x))
    basis_names.append("x*sin(x)")
    basis_funcs.append(lambda x: x * np.cos(x))
    basis_names.append("x*cos(x)")
    basis_funcs.append(lambda x: x * np.exp(-x**2))
    basis_names.append("x*exp(-x^2)")
    basis_funcs.append(lambda x: np.sqrt(np.abs(x)))
    basis_names.append("sqrt|x|")

    n_bases = len(basis_funcs)
    
    # Pre-compute basis matrix
    basis_matrix = np.zeros((n, n_bases))
    for i, func in enumerate(basis_funcs):
        try:
            vals = func(x)
            if not np.all(np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            basis_matrix[:, i] = vals
        except Exception:
            pass

    # Add constant term
    basis_matrix = np.column_stack([np.ones(n), basis_matrix])
    basis_names = ["1"] + basis_names
    n_bases += 1

    # Search for best linear combination of up to 4 basis functions (plus constant)
    # We use a greedy forward selection or random subset search if time permits.
    # Given 200 points, lstsq is fast.
    
    # Start with constant
    best_indices = [0]
    best_coeffs = np.linalg.lstsq(basis_matrix[:, [0]], y, rcond=None)[0]
    best_rss = np.sum((y - basis_matrix[:, [0]] @ best_coeffs)**2)
    
    candidate_indices = list(range(1, n_bases))
    
    # Greedy Forward Selection
    for step in range(1, 6): # Try up to 5 non-constant terms
        if remaining() < 10:
            break
        
        best_add_idx = -1
        best_add_rss = best_rss
        
        for idx in candidate_indices:
            if idx in best_indices:
                continue
            
            trial_indices = best_indices + [idx]
            A = basis_matrix[:, trial_indices]
            
            try:
                coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coeffs
                rss = np.sum((y - pred)**2)
                
                if rss < best_add_rss:
                    best_add_rss = rss
                    best_add_idx = idx
                    best_add_coeffs = coeffs
                    best_add_trial_indices = trial_indices
                    
            except Exception:
                continue
        
        if best_add_idx != -1 and best_add_rss < best_rss:
            best_indices = best_add_trial_indices
            best_coeffs = best_add_coeffs
            best_rss = best_add_rss
            candidate_indices.remove(best_add_idx)
            
            # If we found a near-perfect fit, stop
            if best_rss < 1e-10 * np.var(y):
                break
        else:
            break

    # Store this linear model as a candidate
    if best_rss < np.inf:
        model_type = 'linear_basis'
        model_params = {'indices': best_indices, 'coeffs': best_coeffs, 'basis_funcs': basis_funcs, 'basis_matrix': basis_matrix}
        # Calculate normalized MSE proxy for comparison
        var_y = np.var(y)
        if var_y > 0:
            norm_mse = best_rss / (n * var_y)
        else:
            norm_mse = 0.0 if best_rss < 1e-10 else 1e6
            
        if norm_mse < best_score:
            best_score = norm_mse
            best_model = model_type
            best_model_params = model_params

    # 2. Try specific Non-Linear Forms
    # These require curve_fit and might capture functions not representable by the linear basis above
    # (e.g. a * sin(b*x + c) + d, a * exp(b*x) + c)
    
    nonlinear_forms = [
        (lambda x, a, b, c: a * np.sin(b * x) + c, "a*sin(b*x)+c"),
        (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, "a*sin(b*x+c)+d"),
        (lambda x, a, b, c: a * np.cos(b * x) + c, "a*cos(b*x)+c"),
        (lambda x, a, b, c, d: a * np.cos(b * x + c) + d, "a*cos(b*x+c)+d"),
        (lambda x, a, b, c: a * np.exp(b * x) + c, "a*exp(b*x)+c"),
        (lambda x, a, b, c: a * np.exp(-b * x) + c, "a*exp(-b*x)+c"),
        (lambda x, a, b, c: a * np.exp(-b * x**2) + c, "a*exp(-b*x^2)+c"),
        (lambda x, a, b, c: a / (b + x) + c, "a/(b+x)+c"),
        (lambda x, a, b, c: a * np.log(b + x) + c, "a*log(b+x)+c"),
        (lambda x, a, b, c: a * np.tanh(b * x) + c, "a*tanh(b*x)+c"),
        (lambda x, a, b, c: a * x**b + c, "a*x^b+c"),
        (lambda x, a, b, c: a * x + b * np.sin(x) + c, "linear+sin"),
        (lambda x, a, b, c: a * x + b * np.cos(x) + c, "linear+cos"),
    ]
    
    # Provide multiple initial guesses to help curve_fit
    # Guesses based on data range and magnitude
    y_mean = np.mean(y)
    y_std = np.std(y)
    x_range = np.max(x) - np.min(x)
    
    # Generate a few initial guesses
    guess_sets = [
        [1, 1, 0],
        [y_std, 1, y_mean],
        [y_std, 2, y_mean],
        [y_std, 0.5, y_mean],
        [y_std, 1, 0],
        [1, 1, y_mean],
    ]
    
    for func, name in nonlinear_forms:
        if remaining() < 5:
            break
            
        # Determine number of parameters
        import inspect
        sig = inspect.signature(func)
        n_params = len(sig.parameters) - 1 # Exclude x
        
        # Adjust guess sets to match param count
        valid_guesses = []
        for g in guess_sets:
            if len(g) == n_params:
                valid_guesses.append(g)
        
        if not valid_guesses:
            # Create a generic guess
            if n_params == 3:
                valid_guesses = [[1, 1, 0], [y_std, 1, y_mean]]
            elif n_params == 4:
                valid_guesses = [[1, 1, 0, 0], [y_std, 1, 0, y_mean]]
            else:
                valid_guesses = [[1]*n_params]
        
        best_nl_rss = np.inf
        best_nl_params = None
        
        for p0 in valid_guesses:
            try:
                # Bounds: loose bounds to allow search, but prevent overflow
                # We don't know the exact bounds, so let curve_fit decide or use wide bounds
                # However, for stability, we might want to constrain some parameters
                # For now, try without strict bounds first, catch exceptions
                popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=1000)
                pred = func(x, *popt)
                rss = np.sum((y - pred)**2)
                
                if rss < best_nl_rss:
                    best_nl_rss = rss
                    best_nl_params = popt
                    
            except Exception:
                continue
        
        if best_nl_rss < np.inf:
            if best_nl_rss < best_rss:
                # This nonlinear model is better than our best linear basis model
                model_type = 'nonlinear'
                model_params = {'func': func, 'params': best_nl_params}
                
                var_y = np.var(y)
                if var_y > 0:
                    norm_mse = best_nl_rss / (n * var_y)
                else:
                    norm_mse = 0.0 if best_nl_rss < 1e-10 else 1e6
                
                if norm_mse < best_score:
                    best_score = norm_mse
                    best_model = model_type
                    best_model_params = model_params

    # 3. Fallback: PCHIP Interpolation
    # If no parametric model fits well, or if we just want a robust fallback
    # PCHIP is better than CubicSpline for non-smooth data
    try:
        pchip = PchipInterpolator(x_unique, y_unique)
        pred_pchip = pchip(x_unique)
        rss_pchip = np.sum((y_unique - pred_pchip)**2)
        
        var_y = np.var(y)
        if var_y > 0:
            norm_mse_pchip = rss_pchip / (len(y_unique) * var_y)
        else:
            norm_mse_pchip = 0.0 if rss_pchip < 1e-10 else 1e6
        
        if norm_mse_pchip < best_score:
            best_score = norm_mse_pchip
            best_model = 'pchip'
            best_model_params = {'interpolator': pchip}
            
    except Exception:
        pass

    # 4. Construct Predictor
    
    if best_model == 'linear_basis':
        indices = best_model_params['indices']
        coeffs = best_model_params['coeffs']
        basis_funcs = best_model_params['basis_funcs']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                A_pred = np.zeros((len(xi), len(indices)))
                for col_idx, b_idx in enumerate(indices):
                    if b_idx == 0:
                        A_pred[:, col_idx] = 1.0
                    else:
                        # Note: basis_funcs in best_model_params does NOT include the constant at index 0
                        # Wait, in the search loop, I added constant to basis_matrix but basis_funcs list was 1-indexed relative to the list?
                        # Let's check: 
                        # basis_funcs list starts with x^1...
                        # basis_matrix was column_stack([ones, basis_matrix_original])
                        # So index 0 in basis_matrix is constant.
                        # Index i in basis_matrix (i>0) corresponds to basis_funcs[i-1].
                        
                        # My best_indices comes from the search on basis_matrix.
                        # So if b_idx == 0, it's constant.
                        # If b_idx > 0, it's basis_funcs[b_idx - 1].
                        
                        vals = basis_funcs[b_idx - 1](xi)
                        if not np.all(np.isfinite(vals)):
                            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                        A_pred[:, col_idx] = vals
                preds = A_pred @ coeffs
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

    elif best_model == 'pchip':
        interpolator = best_model_params['interpolator']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = interpolator(xi)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))
                
    else:
        # Should not happen, but return mean
        mean_y = np.mean(y)
        def predict(X_in):
            return np.full(len(X_in), mean_y)

    return predict
