import time
import numpy as np
from scipy.interpolate import CubicSpline
import itertools

def fit(X_train, y_train):
    """
    Fit a regression model.
    Strategy:
    1. Try Polynomial Regression (degrees 0-15) to find exact or near-exact fit.
    2. If not perfect, try Greedy Basis Search with rich function library.
    3. If still not perfect, fallback to Cubic Spline Interpolation.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 320.0

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

    # Sort by x for spline fallback
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    # --- Strategy 1: Polynomial Regression ---
    best_poly_degree = -1
    best_poly_coeffs = None
    best_poly_rss = np.inf

    # Try degrees 0 to 15
    for deg in range(0, 16):
        if remaining() < 2:
            break
        try:
            # Use np.polyfit or lstsq for stability
            # We use lstsq on Vandermonde matrix for better control
            V = np.vander(x, N=deg+1, increasing=True)
            coeffs, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
            
            if not np.all(np.isfinite(coeffs)):
                continue
                
            pred = V @ coeffs
            rss = np.sum((y - pred) ** 2)
            
            if rss < best_poly_rss:
                best_poly_rss = rss
                best_poly_coeffs = coeffs
                best_poly_degree = deg
                
            # If we find a near-perfect fit, stop early
            if rss < 1e-10:
                break
        except Exception:
            continue

    # --- Strategy 2: Basis Search (Greedy) ---
    # Only if polynomial didn't give a near-perfect fit
    use_basis = False
    best_indices = None
    best_coeffs = None
    best_rss = np.inf

    if best_poly_rss > 1e-10 and remaining() > 10:
        basis_funcs = []
        basis_names = []
        
        # Constant
        basis_funcs.append(lambda x: np.ones_like(x))
        basis_names.append("1")
        
        # Polynomials (up to 5)
        for p in range(1, 6):
            basis_funcs.append(lambda x, p=p: x**p)
            basis_names.append(f"x^{p}")
            
        # Trigonometric
        for k in [1, 2, 3]:
            basis_funcs.append(lambda x, k=k: np.sin(k * x))
            basis_names.append(f"sin({k}x)")
            basis_funcs.append(lambda x, k=k: np.cos(k * x))
            basis_names.append(f"cos({k}x)")
            
        # Hyperbolic
        basis_funcs.append(lambda x: np.tanh(x))
        basis_names.append("tanh(x)")
        
        # Exponential
        basis_funcs.append(lambda x: np.exp(np.clip(x, -10, 10)))
        basis_names.append("exp(x)")
        basis_funcs.append(lambda x: np.exp(np.clip(-x, -10, 10)))
        basis_names.append("exp(-x)")
        
        # Logarithmic
        basis_funcs.append(lambda x: np.log(np.abs(x) + 1e-12))
        basis_names.append("log|x|")
        
        # Rational
        basis_funcs.append(lambda x: 1.0 / (x + 1e-12))
        basis_names.append("1/x")
        basis_funcs.append(lambda x: 1.0 / (1.0 + x**2))
        basis_names.append("1/(1+x^2)")
        
        n_bases = len(basis_funcs)
        
        # Pre-compute design matrix
        basis_matrix = np.zeros((n, n_bases))
        for i, func in enumerate(basis_funcs):
            try:
                vals = func(x)
                if not np.all(np.isfinite(vals)):
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                basis_matrix[:, i] = vals
            except Exception:
                pass
        
        # Greedy Forward Selection
        selected = [0] # Start with constant
        current_rss = np.inf
        
        # Calculate initial RSS with just constant
        try:
            A_init = basis_matrix[:, [0]]
            c_init, _, _, _ = np.linalg.lstsq(A_init, y, rcond=None)
            current_rss = np.sum((y - A_init @ c_init)**2)
        except:
            pass

        max_features = 5
        for _ in range(max_features):
            if remaining() < 5:
                break
                
            best_candidate_idx = -1
            best_candidate_rss = current_rss
            
            for i in range(n_bases):
                if i in selected:
                    continue
                
                new_selected = selected + [i]
                A = basis_matrix[:, new_selected]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    if not np.all(np.isfinite(coeffs)):
                        continue
                    pred = A @ coeffs
                    rss = np.sum((y - pred)**2)
                    
                    if rss < best_candidate_rss:
                        best_candidate_rss = rss
                        best_candidate_idx = i
                except Exception:
                    continue
            
            if best_candidate_idx == -1:
                break
                
            selected.append(best_candidate_idx)
            current_rss = best_candidate_rss
            
            if current_rss < 1e-10:
                break
        
        if current_rss < 1e-10:
            use_basis = True
            best_indices = selected
            A = basis_matrix[:, selected]
            best_coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            best_rss = current_rss
        elif current_rss < best_poly_rss * 0.1:
            # If basis is significantly better than poly, use it
            use_basis = True
            best_indices = selected
            A = basis_matrix[:, selected]
            best_coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            best_rss = current_rss

    # --- Strategy 3: Cubic Spline Fallback ---
    use_spline = False
    spline = None
    
    if not use_basis and best_poly_rss > 1e-10:
        if len(x_unique) >= 4:
            try:
                spline = CubicSpline(x_unique, y_unique, bc_type='natural')
                use_spline = True
            except Exception:
                use_spline = False

    # --- Construct Predictor ---
    median_y = np.median(y)
    
    if use_basis and best_coeffs is not None:
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
                
    elif best_poly_degree != -1 and best_poly_coeffs is not None:
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                # Evaluate polynomial
                # np.polyval expects highest degree first, but we used increasing=True in vander
                # So coeffs are [c0, c1, ..., cN]
                # We can just do sum(c_i * x^i)
                preds = np.zeros(len(xi))
                for i, c in enumerate(best_poly_coeffs):
                    preds += c * (xi ** i)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, median_y)
                return preds
            except Exception:
                return np.full(len(X_in), median_y)
                
    elif use_spline:
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
        # Fallback: Mean
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            return np.full(len(X_in), median_y)

    return predict
