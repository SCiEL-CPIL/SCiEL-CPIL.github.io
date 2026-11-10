"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_ridge_poly(x, y, degree=5, alpha=1e-4):
    """Fit a degree-`degree` polynomial via ridge regression in the monomial basis."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < degree + 1:
        # Fall back to linear if not enough points
        if n >= 2:
            c = np.zeros(degree + 1)
            c[:2] = np.polyfit(x, y, 1)[::-1]
            return c
        return np.zeros(degree + 1)
    
    # Build Vandermonde matrix in ascending powers
    V = np.vander(x, N=degree + 1, increasing=True)
    
    # Ridge: minimize ||V c - y||^2 + alpha * ||c||^2
    # Solution: (V^T V + alpha I) c = V^T y
    A = V.T @ V + alpha * np.eye(degree + 1)
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Ensure finite
    if not np.all(np.isfinite(c)):
        c = np.zeros(degree + 1)
        c[:2] = np.polyfit(x, y, 1)[::-1]
    
    return c


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-five polynomial to the unresolved subgrid tendency."""
    t_start = time.time()
    deadline = t_start + 120.0  # safety margin
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Try a small set of regularization strengths and pick the best via
    # leave-one-out cross-validation (approximated by ridge LOO formula for speed)
    # LOO MSE for ridge: (y_i - f_i)^2 / (1 - h_ii)^2 where h_ii = V_i (V^T V + alpha I)^{-1} V_i^T
    # For efficiency with large n, use the hat-matrix approach only if n is small enough.
    
    n = len(x)
    
    # If n is very large, just use a fixed small alpha
    if n > 50000:
        alpha = 1e-3
        c = _fit_ridge_poly(x, y, degree=5, alpha=alpha)
        return c
    
    # Try a few alphas and use LOO-CV to select
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    best_c = None
    best_loo_mse = np.inf
    
    # Precompute V
    V = np.vander(x, N=6, increasing=True)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    x_centered = x - np.mean(x)
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        try:
            A = V.T @ V + alpha * np.eye(6)
            b = V.T @ y
            c = np.linalg.solve(A, b)
            
            # Compute LOO-CV MSE efficiently
            # h = V (V^T V + alpha I)^{-1} V^T
            # This is expensive for large n; use the identity:
            # LOO residual = (y_i - f_i) / (1 - h_ii)
            # where h_ii = V_i^T (V^T V + alpha I)^{-1} V_i
            # We can compute h_ii for all i as diag(V @ inv(A) @ V.T)
            # But that's O(n^2 * 6^3) which is fine for n up to ~50k
            
            # Actually, let's just do a simple K-fold CV for speed
            # Or use the analytic LOO formula
            inv_A = np.linalg.inv(A)
            h_diag = np.einsum('ij,jk,ik->i', V, inv_A, V)
            f = V @ c
            residuals = y - f
            loo_resid = residuals / (1.0 - h_diag)
            loo_mse = np.mean(loo_resid ** 2)
            
            if loo_mse < best_loo_mse:
                best_loo_mse = loo_mse
                best_c = c.copy()
        except Exception:
            continue
    
    if best_c is None:
        # Fallback: least squares with small ridge
        best_c = _fit_ridge_poly(x, y, degree=5, alpha=1e-4)
    
    # Ensure finite
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(6)
        best_c[:2] = np.polyfit(x, y, 1)[::-1]
    
    return best_c
# EVOLVE-BLOCK-END
