import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import itertools

def fit(X_train, y_train):
    """
    Fit a regression model. Primary strategy: cubic spline interpolation on sorted
    1D noiseless data (exact reconstruction between points). Fallback: basis search.
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
    
    # Check if data is strictly increasing in x (needed for unique interpolation)
    # Allow small duplicates by using np.unique with return_index
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    
    # Strategy 1: Cubic Spline Interpolation (primary for noiseless 1D smooth data)
    use_spline = False
    spline = None
    
    if len(x_unique) >= 4:  # Need at least 4 points for cubic spline
        try:
            y_unique = y_sorted[unique_indices]
            # Fit cubic spline
            spline = CubicSpline(x_unique, y_unique, bc_type='natural')
            use_spline = True
        except Exception:
            use_spline = False
    
    # Strategy 2: Basis search (fallback)
    best_aic = np.inf
    best_indices = None
    best_coeffs = None
    best_rss = np.inf
    
    if not use_spline or remaining() > 10:
        # Only do basis search if spline failed or we have ample time
        basis_funcs = []
        basis_names = []
        
        basis_funcs.append(lambda x: np.ones_like(x))
        basis_names.append("1")
        
        for p in range(1, 7):
            basis_funcs.append(lambda x, p=p: x**p)
            basis_names.append(f"x^{p}")
            
        for k in [1, 2, 3, 4]:
            basis_funcs.append(lambda x, k=k: np.sin(k * x))
            basis_names.append(f"sin({k}x)")
            basis_funcs.append(lambda x, k=k: np.cos(k * x))
            basis_names.append(f"cos({k}x)")
            
        basis_funcs.append(lambda x: np.tanh(x))
        basis_names.append("tanh(x)")
        basis_funcs.append(lambda x: np.sinh(x))
        basis_names.append("sinh(x)")
        basis_funcs.append(lambda x: np.cosh(x))
        basis_names.append("cosh(x)")
        
        basis_funcs.append(lambda x: np.exp(np.clip(x, -10, 10)))
        basis_names.append("exp(x)")
        basis_funcs.append(lambda x: np.exp(np.clip(-x, -10, 10)))
        basis_names.append("exp(-x)")
        
        basis_funcs.append(lambda x: np.log(np.abs(x) + 1e-12))
        basis_names.append("log|x|")
        basis_funcs.append(lambda x: np.log(x + 10))
        basis_names.append("log(x+10)")
        
        basis_funcs.append(lambda x: 1.0 / (x + 1e-12))
        basis_names.append("1/x")
        basis_funcs.append(lambda x: 1.0 / (1.0 + x**2))
        basis_names.append("1/(1+x^2)")
        
        basis_funcs.append(lambda x: np.sqrt(np.abs(x)))
        basis_names.append("sqrt|x|")
        
        n_bases = len(basis_funcs)
        
        basis_matrix = np.zeros((n, n_bases))
        for i, func in enumerate(basis_funcs):
            try:
                vals = func(x)
                if not np.all(np.isfinite(vals)):
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                basis_matrix[:, i] = vals
            except Exception:
                pass
        
        fixed_indices = [0]
        candidate_indices = list(range(1, n_bases))
        max_additional = 4
        
        for k_add in range(1, max_additional + 1):
            if remaining() < 5:
                break
            
            for combo in itertools.combinations(candidate_indices, k_add):
                if remaining() < 2:
                    break
                    
                indices = fixed_indices + list(combo)
                A = basis_matrix[:, indices]
                
                try:
                    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
                    
                    if not np.all(np.isfinite(coeffs)):
                        continue
                        
                    pred = A @ coeffs
                    rss = np.sum((y - pred) ** 2)
                    
                    if rss < 1e-12:
                        best_aic = -np.inf
                        best_indices = indices
                        best_coeffs = coeffs
                        best_rss = rss
                        break
                    
                    k_params = len(indices)
                    aic = n * np.log(rss / n) + 2 * k_params
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_indices = indices
                        best_coeffs = coeffs
                        best_rss = rss
                        
                except Exception:
                    continue
            
            if best_rss < 1e-12:
                break
        
        if best_coeffs is None:
            best_indices = [0]
            A = basis_matrix[:, [0]]
            best_coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            best_rss = np.sum((y - A @ best_coeffs)**2)
    
    # Construct predictor
    median_y = np.median(y)
    
    if use_spline and spline is not None:
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
    else:
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                A_pred = np.zeros((len(xi), len(best_indices)))
                for col_idx, b_idx in enumerate(best_indices):
                    try:
                        vals = basis_funcs[b_idx](xi)
                        if not np.all(np.isfinite(vals)):
                            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                        A_pred[:, col_idx] = vals
                    except Exception:
                        A_pred[:, col_idx] = 0.0
                preds = A_pred @ best_coeffs
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, median_y)
                return preds
            except Exception:
                return np.full(len(X_in), median_y)
    
    return predict
