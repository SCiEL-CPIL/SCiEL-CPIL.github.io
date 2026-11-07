import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (global) + 5 Gaussian RBFs (local).
    
    The Chebyshev terms provide a stable global approximation.
    The Gaussian RBFs provide local flexibility to capture non-polynomial features.
    
    5 Chebyshev terms (T0..T4) on mapped domain t = x/3 in [-1,1]
    5 Gaussian RBFs centered at -2.4, -1.2, 0, 1.2, 2.4 with width sigma=1.0
    
    Total: 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Chebyshev polynomials T0..T4 via recurrence
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Gaussian RBFs with centers spread across the domain
    # 5 Gaussians at -2.4, -1.2, 0, 1.2, 2.4
    sigma_sq = 2.0 * 1.0**2  # 2.0
    
    centers = np.array([-2.4, -1.2, 0.0, 1.2, 2.4])
    diffs = x[None, :] - centers[:, None]  # shape (5, n)
    gaussians = np.exp(-diffs**2 / sigma_sq)  # shape (5, n)
    
    # Linear combination: 5 Chebyshev + 5 Gaussian = 10 terms
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * gaussians[0] +
         params[6] * gaussians[1] +
         params[7] * gaussians[2] +
         params[8] * gaussians[3] +
         params[9] * gaussians[4])
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
