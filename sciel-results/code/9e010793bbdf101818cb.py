import time
import numpy as np
from scipy.interpolate import CubicSpline
import itertools

def fit(X_train, y_train):
    """
    Fit a regression model.
    Primary strategy: Exact interpolation via cubic spline on sorted 1D noiseless data.
    Fallback strategy: Basis search with AIC-based model selection.
    Returns a predict callable.
    """
    start_time = time.time()
    # 330s limit. Use 320s deadline to be safe.
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

    # Sort by x for interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check if data is strictly increasing in x (needed for unique interpolation)
    # Allow small duplicates by using np.unique with return_index
    # Note: If there are duplicate x values with different y values, the data is not a function.
    # However, the task implies "equations", so x should map to a single y.
    # If duplicates exist, we average them or take the first. np.unique with return_index takes the first.
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    
    # Strategy 1: Cubic Spline Interpolation (primary for noiseless 1D smooth data)
    # Since data is noiseless and 1D, if the function is smooth, spline interpolation 
    # between the 200 points should be extremely accurate, often exact if the function 
    # is a spline of that degree, or very close otherwise. 
    # Given 200 points, cubic spline is a strong baseline.
    
    use_spline = False
    spline = None
    
    if len(x_unique) >= 4:
        try:
            y_unique = y_sorted[unique_indices]
            # Use 'natural' boundary conditions for stability
            spline = CubicSpline(x_unique, y_unique, bc_type='natural')
            use_spline = True
        except Exception:
            use_spline = False

    # Strategy 2: Basis search (fallback)
    # If spline fails (e.g., non-smooth function, or if we detect it's a simple parametric form),
    # we try to fit a combination of basis functions.
    
    best_aic = np.inf
    best_indices = None
    best_coeffs = None
    best_rss = np.inf
    
    # We only enter this if spline didn't work OR if we have plenty of time and want to check 
    # if a simpler global model exists (though spline is usually better for interpolation).
    # However, if the function is e.g. sin(x), spline is great. If it's a polynomial, spline is great.
    # The main risk of spline is extrapolation outside the training range.
    # Test points are in the same range as training points (uniform distribution over the same interval).
    # So extrapolation is not an issue.
    
    # Let's still run a quick basis search to see if we can find an exact fit with fewer parameters,
    # which might be more robust if there's any slight noise or if the spline has boundary artifacts.
    # But the prompt says "noiseless". So spline should be nearly perfect.
    # The main advantage of basis search is if the function is exactly one of the bases.
    
    # Let's do a light basis search if time permits.
    if not use_spline or remaining() > 10:
        
        basis_funcs = []
        basis_names = []
        
        # Constant
        basis_funcs.append(lambda x: np.ones_like(x))
        basis_names.append("1")
        
        # Polynomials
        for p in range(1, 6):
            basis_funcs.append(lambda x, p=p: x**p)
            basis_names.append(f"x^{p}")
            
        # Trigonometric
        for k in [1, 2, 3, 4]:
            basis_funcs.append(lambda x, k=k: np.sin(k * x))
            basis_names.append(f"sin({k}x)")
            basis_funcs.append(lambda x, k=k: np.cos(k * x))
            basis_names.append(f"cos({k}x)")
            
        # Hyperbolic
        basis_funcs.append(lambda x: np.tanh(x))
        basis_names.append("tanh(x)")
        basis_funcs.append(lambda x: np.sinh(x))
        basis_names.append("sinh(x)")
        basis_funcs.append(lambda x: np.cosh(x))
        basis_names.append("cosh(x)")
        
        # Exponential
        basis_funcs.append(lambda x: np.exp(np.clip(x, -10, 10)))
        basis_names.append("exp(x)")
        basis_funcs.append(lambda x: np.exp(np.clip(-x, -10, 10)))
        basis_names.append("exp(-x)")
        
        # Logarithmic
        basis_funcs.append(lambda x: np.log(np.abs(x) + 1e-12))
        basis_names.append("log|x|")
        basis_funcs.append(lambda x: np.log(x + 10))
        basis_names.append("log(x+10)")
        
        # Rational
        basis_funcs.append(lambda x: 1.0 / (x + 1e-12))
        basis_names.append("1/x")
        basis_funcs.append(lambda x: 1.0 / (1.0 + x**2))
        basis_names.append("1/(1+x^2)")
        
        # Sqrt
        basis_funcs.append(lambda x: np.sqrt(np.abs(x)))
        basis_names.append("sqrt|x|")
        
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
        
        fixed_indices = [0] # Always include constant
        candidate_indices = list(range(1, n_bases))
        max_additional = 3 # Keep it small to ensure speed
        
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
                    
                    # If we find a near-perfect fit, record it
                    if rss < 1e-10:
                        best_aic = -np.inf
                        best_indices = indices
                        best_coeffs = coeffs
                        best_rss = rss
                        break
                    
                    k_params = len(indices)
                    aic = n * np.log(rss / n + 1e-15) + 2 * k_params
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_indices = indices
                        best_coeffs = coeffs
                        best_rss = rss
                        
                except Exception:
                    continue
            
            if best_rss < 1e-10:
                break
        
        # If spline failed but basis search found something, use basis
        # If spline succeeded, we generally prefer spline for interpolation accuracy unless basis is perfect
        # Actually, if basis is perfect (rss ~ 0), it's better.
        # If spline is "good" but basis is "bad", use spline.
        # How to decide? 
        # Let's calculate the training error of the spline.
        if use_spline:
            try:
                spline_pred = spline(x_unique)
                spline_rss = np.sum((y_unique - spline_pred)**2)
            except:
                spline_rss = np.inf
        
        # Prefer basis if it has significantly lower RSS or if spline failed
        use_basis = False
        if not use_spline:
            if best_coeffs is not None:
                use_basis = True
        else:
            # If basis found a perfect fit, use it
            if best_rss < 1e-10 and spline_rss > 1e-10:
                use_basis = True
            # Or if basis RSS is much better than spline RSS (unlikely for noiseless smooth data)
            elif best_rss < spline_rss * 0.1 and best_rss < 1e-5:
                use_basis = True
        # If both are bad, fallback to constant or mean
        if best_coeffs is None and not use_spline:
            # Fallback to mean
            mean_y = np.mean(y)
            def predict(X_in):
                X_in = np.asarray(X_in, dtype=np.float64)
                return np.full(len(X_in), mean_y)
            return predict

    # Construct predictor
    median_y = np.median(y)
    
    if use_spline and not use_basis:
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
    elif use_basis and best_coeffs is not None:
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
    else:
        # Fallback: If something went wrong, return median
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            return np.full(len(X_in), median_y)

    return predict
