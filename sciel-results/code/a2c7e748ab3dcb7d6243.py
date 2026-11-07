"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    def elapsed_ok():
        return (time.time() - start_time) < deadline

    # ---------- Feature generators (polynomial, trig, exp, log) ----------
    def poly_basis(X, deg=4):
        x = X[:, 0]
        cols = [np.ones_like(x)]
        p = np.ones_like(x)
        for k in range(1, deg + 1):
            p = p * x
            cols.append(p)
        return np.column_stack(cols)

    def trig_basis(X, n_harm=6):
        x = X[:, 0]
        cols = [np.ones_like(x)]
        for k in range(1, n_harm + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        return np.column_stack(cols)

    def exp_basis(X, n_terms=6):
        x = X[:, 0]
        # Use scaled x to avoid overflow; range roughly [-3,3] or [0.1,6]
        # We'll fit coefficients for exp(k*x/scale)
        cols = [np.ones_like(x)]
        for k in range(1, n_terms + 1):
            # Use a moderate scaling to keep values in check
            cols.append(np.exp(k * 0.5 * x))
        return np.column_stack(cols)

    def log_basis(X):
        x = X[:, 0]
        # Only valid for x > 0; for negative x, we can't use log directly.
        # We'll handle this by checking the range.
        cols = [np.ones_like(x)]
        if np.min(x) > 0:
            cols.append(np.log(x))
            cols.append(np.log(x) * x)
            cols.append(np.log(x) ** 2)
        else:
            # Fallback: use log(abs(x)) + sign? No, just skip or use poly
            pass
        return np.column_stack(cols)

    def combined_basis(X, deg=5, n_harm=4):
        # Combine polynomial and trigonometric features
        pb = poly_basis(X, deg)
        tb = trig_basis(X, n_harm)
        return np.column_stack([pb, tb])

    def rbf_basis(X, centers=None, width=1.0):
        x = X[:, 0]
        if centers is None:
            # Use quantiles as centers
            centers = np.linspace(np.min(x), np.max(x), 15)
        cols = [np.ones_like(x)]
        for c in centers:
            cols.append(np.exp(-((x - c) ** 2) / (2 * width ** 2)))
        return np.column_stack(cols)

    # ---------- Model fitting helper ----------
    def fit_model(basis_func, max_iter=100):
        if not elapsed_ok():
            return None, None, float('inf')
        
        try:
            B = basis_func(X)
            # Check for NaN/Inf
            if not np.all(np.isfinite(B)):
                return None, None, float('inf')
            
            # Use least squares
            w, residuals, rank, s = np.linalg.lstsq(B, y, rcond=None)
            pred = B @ w
            mse = np.mean((pred - y) ** 2)
            return w, B, mse
        except:
            return None, None, float('inf')

    # ---------- Nonlinear fit for exp/log models ----------
    def fit_nonlinear(basis_func, init_params=None, max_iter=50):
        if not elapsed_ok():
            return None, None, float('inf')
        
        try:
            x = X[:, 0]
            
            def model(params):
                if len(params) == 1:
                    # Pure exp: y = a * exp(b * x)
                    a, b = params[0], params[1] if len(params) > 1 else 0
                    return a * np.exp(b * x)
                elif len(params) == 2:
                    # y = a * exp(b * x)
                    a, b = params
                    return a * np.exp(b * x)
                elif len(params) == 3:
                    # y = a * log(x) + b * x + c (for x > 0)
                    a, b, c = params
                    return a * np.log(x) + b * x + c
                elif len(params) == 4:
                    # y = a * x^b + c * x + d
                    a, b, c, dd = params
                    return a * np.power(x, b) + c * x + dd
                return np.zeros_like(x)
            
            # This is getting complicated. Let's stick to linear basis expansion
            # which is more robust.
            return None, None, float('inf')
        except:
            return None, None, float('inf')

    # ---------- Candidate models ----------
    candidates = []
    
    # 1. Polynomial degree 2-8
    for deg in range(2, 9):
        if not elapsed_ok():
            break
        w, B, mse = fit_model(lambda X: poly_basis(X, deg))
        if w is not None and mse < float('inf'):
            candidates.append((mse, w, B, lambda X, w=w, deg=deg: poly_basis(X, deg) @ w))
    
    # 2. Trigonometric
    for nh in range(2, 8):
        if not elapsed_ok():
            break
        w, B, mse = fit_model(lambda X: trig_basis(X, nh))
        if w is not None and mse < float('inf'):
            candidates.append((mse, w, B, lambda X, w=w, nh=nh: trig_basis(X, nh) @ w))
    
    # 3. Combined poly+trig
    for deg in range(2, 6):
        for nh in range(1, 4):
            if not elapsed_ok():
                break
            w, B, mse = fit_model(lambda X: combined_basis(X, deg, nh))
            if w is not None and mse < float('inf'):
                candidates.append((mse, w, B, lambda X, w=w, deg=deg, nh=nh: combined_basis(X, deg, nh) @ w))
    
    # 4. RBF with different widths
    for width in [0.5, 1.0, 2.0, 3.0]:
        if not elapsed_ok():
            break
        w, B, mse = fit_model(lambda X: rbf_basis(X, width=width))
        if w is not None and mse < float('inf'):
            candidates.append((mse, w, B, lambda X, w=w, width=width: rbf_basis(X, width=width) @ w))
    
    # 5. Exponential basis (linear in coefficients)
    for n_exp in range(2, 6):
        if not elapsed_ok():
            break
        w, B, mse = fit_model(lambda X: exp_basis(X, n_exp))
        if w is not None and mse < float('inf'):
            candidates.append((mse, w, B, lambda X, w=w, n_exp=n_exp: exp_basis(X, n_exp) @ w))

    # 6. Log basis (if x > 0)
    if np.min(X[:, 0]) > 0:
        w, B, mse = fit_model(log_basis)
        if w is not None and mse < float('inf'):
            candidates.append((mse, w, B, lambda X: log_basis(X) @ w))

    # If no candidates, use simple linear
    if not candidates:
        w, B, mse = fit_model(lambda X: poly_basis(X, 1))
        if w is not None:
            candidates.append((mse, w, B, lambda X: poly_basis(X, 1) @ w))
        else:
            # Fallback: constant
            c = np.mean(y)
            return lambda X: np.full(len(X), c)

    # Select best candidate
    candidates.sort(key=lambda x: x[0])
    best_mse, best_w, best_B, best_predict = candidates[0]
    
    # Optional: try to refine with higher-degree poly if time permits
    if elapsed_ok() and best_mse > 1e-10:
        # Try a few more specific models
        for deg in range(9, 12):
            if not elapsed_ok():
                break
            w, B, mse = fit_model(lambda X: poly_basis(X, deg))
            if w is not None and mse < best_mse:
                best_mse = mse
                best_w = w
                best_B = B
                best_predict = lambda X, w=w, deg=deg: poly_basis(X, deg) @ w
    
    return best_predict
# EVOLVE-BLOCK-END
