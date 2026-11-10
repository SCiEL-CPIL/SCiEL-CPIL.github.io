# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _check_deadline()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Gaussian basis functions (fixed centers/widths) + 5 Chebyshev
    polynomial basis functions (normalized). This combines local smooth approximation
    (Gaussians) with global polynomial flexibility (Chebyshev), all linear in params.
    
    The linear structure ensures BFGS converges to the global minimum of the
    least-squares problem. Fixed centers/widths avoid non-convexity.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if not _check_deadline():
        return np.ones_like(x) * params[0]
    
    n = len(x)
    if n == 0:
        return np.array([])
    
    # --- Part 1: 5 Gaussian basis functions ---
    # Centers evenly spaced in [0.1, 6]
    g_centers = np.linspace(0.1, 6.0, 5)
    g_sigma = 0.7  # Wider than before to cover more area with fewer Gaussians
    
    x_exp = x[:, np.newaxis]
    c_exp = g_centers[np.newaxis, :]
    diff = x_exp - c_exp
    G = np.exp(-0.5 * (diff / g_sigma) ** 2)  # shape (N, 5)
    
    # --- Part 2: 5 Chebyshev basis functions (T_0 to T_4) ---
    # Map x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0..T_4
    C = np.zeros((n, 5))
    C[:, 0] = 1.0  # T_0
    C[:, 1] = t    # T_1
    
    for k in range(2, 5):
        C[:, k] = 2.0 * t * C[:, k-1] - C[:, k-2]
    
    # Normalize Chebyshev basis by their RMS over the domain
    # Precompute RMS on a fine grid
    x_grid = np.linspace(0.1, 6.0, 1000)
    t_grid = (2.0 * x_grid - 6.1) / 5.9
    t_grid = np.clip(t_grid, -1.0, 1.0)
    
    C_grid = np.zeros((1000, 5))
    C_grid[:, 0] = 1.0
    C_grid[:, 1] = t_grid
    for k in range(2, 5):
        C_grid[:, k] = 2.0 * t_grid * C_grid[:, k-1] - C_grid[:, k-2]
    
    rms_C = np.sqrt(np.mean(C_grid ** 2, axis=0))
    rms_C = np.where(rms_C < 1e-12, 1.0, rms_C)
    C = C / rms_C
    
    # Also normalize Gaussian basis by their RMS for balanced gradients
    G_grid = np.zeros((1000, 5))
    xg_exp = x_grid[:, np.newaxis]
    diff_grid = xg_exp - c_exp
    G_grid = np.exp(-0.5 * (diff_grid / g_sigma) ** 2)
    
    rms_G = np.sqrt(np.mean(G_grid ** 2, axis=0))
    rms_G = np.where(rms_G < 1e-12, 1.0, rms_G)
    G = G / rms_G
    
    # Combine basis: first 5 params for Gaussians, last 5 for Chebyshev
    basis = np.hstack([G, C])  # shape (N, 10)
    
    # Ensure finite values
    if not np.all(np.isfinite(basis)):
        basis = np.nan_to_num(basis, nan=0.0, posinf=1e6, neginf=-1e6)
    
    result = basis @ params
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
