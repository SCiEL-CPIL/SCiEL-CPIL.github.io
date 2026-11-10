"""Rewritten program: shared degree-five polynomial fit with optimized regularization search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _ridge_solve(V, y, lam, col_scales):
    """Solve ridge regression with scale-aware regularization.
    
    Returns coefficient vector of length len(col_scales), or None on failure.
    """
    n = len(col_scales)
    if lam <= 0.0:
        # Unregularized least squares
        try:
            c, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            if not np.all(np.isfinite(c)):
                return None
            return c
        except Exception:
            return None
    
    # Scale-aware ridge: min ||Vc - y||^2 + lam * sum((col_scales_i * c_i)^2)
    # This is equivalent to: min ||Vc - y||^2 + (V @ diag(col_scales))^T (V @ diag(col_scales)) * lam
    # Or in standard form: (V^T V + lam * diag(col_scales^2)) c = V^T y
    D = col_scales ** 2
    reg_diag = lam * D
    reg_diag[0] = 0.0  # Don't penalize intercept
    
    try:
        A = V.T @ V + np.diag(reg_diag)
        b = V.T @ y
        c = np.linalg.solve(A, b)
        if not np.all(np.isfinite(c)):
            return None
        return c
    except Exception:
        return None


def _cv_mse(x, y, deg, lam, n_folds=5, seed=42):
    """K-fold cross-validation MSE for polynomial degree deg with ridge lambda lam."""
    n = len(x)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        # Not enough data for CV, just fit and evaluate
        V = np.polynomial.polynomial.polyvander(x, deg)
        col_scales = np.std(V, axis=0)
        col_scales[col_scales < 1e-12] = 1.0
        c = _ridge_solve(V, y, lam, col_scales)
        if c is None:
            return np.inf
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        if len(train_idx) < deg + 1:
            continue
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        V_train = np.polynomial.polynomial.polyvander(x_train, deg)
        col_scales = np.std(V_train, axis=0)
        col_scales[col_scales < 1e-12] = 1.0
        
        c = _ridge_solve(V_train, y_train, lam, col_scales)
        if c is None:
            continue
        
        pred = np.polynomial.polynomial.polyval(x_val, c)
        total_mse += np.mean((pred - y_val) ** 2)
        valid_folds += 1
    
    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    best_coeff = None
    best_score = np.inf
    best_deg = 5
    best_lam = 0.0
    
    # Phase 1: Coarse grid search over (degree, lambda)
    # Use log-spaced lambdas including 0
    coarse_lambdas = [0.0] + list(np.logspace(-6, 3, 20))
    degrees = [3, 4, 5]
    
    for deg in degrees:
        for lam in coarse_lambdas:
            if time.time() > deadline - 15.0:
                break
            
            score = _cv_mse(x, y, deg, lam, n_folds=5, seed=42)
            
            if score < best_score:
                best_score = score
                best_deg = deg
                best_lam = lam
    
    if time.time() > deadline - 15.0:
        # Time pressure: fit with best found so far
        V = np.polynomial.polynomial.polyvander(x, best_deg)
        col_scales = np.std(V, axis=0)
        col_scales[col_scales < 1e-12] = 1.0
        c = _ridge_solve(V, y, best_lam, col_scales)
        if c is not None:
            out = np.zeros(6)
            out[:best_deg + 1] = c
            return out
        return np.zeros(6)
    
    # Phase 2: Fine-tune lambda around the best value
    if time.time() < deadline - 20.0:
        if best_lam > 0:
            fine_lambdas = list(np.logspace(np.log10(best_lam) - 0.5, np.log10(best_lam) + 0.5, 15))
        else:
            fine_lambdas = [0.0] + list(np.logspace(-8, -2, 15))
        
        for lam in fine_lambdas:
            if time.time() > deadline - 15.0:
                break
            score = _cv_mse(x, y, best_deg, lam, n_folds=5, seed=42)
            if score < best_score:
                best_score = score
                best_lam = lam
    
    # Phase 3: Try adjacent degrees with the best lambda
    if time.time() < deadline - 10.0:
        for deg in sorted(set([max(2, best_deg - 1), best_deg, min(5, best_deg + 1)])):
            if time.time() > deadline - 8.0:
                break
            score = _cv_mse(x, y, deg, best_lam, n_folds=5, seed=42)
            if score < best_score:
                best_score = score
                best_deg = deg
    
    # Phase 4: Final fit with all data using best (deg, lam)
    V = np.polynomial.polynomial.polyvander(x, best_deg)
    col_scales = np.std(V, axis=0)
    col_scales[col_scales < 1e-12] = 1.0
    c = _ridge_solve(V, y, best_lam, col_scales)
    
    if c is not None and np.all(np.isfinite(c)):
        out = np.zeros(6)
        out[:best_deg + 1] = c
        return out
    
    # Fallback: unregularized degree-5 fit
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 5)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:6] = c
            return out
    except Exception:
        pass
    
    return np.zeros(6)
# EVOLVE-BLOCK-END
