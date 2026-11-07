import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global polynomial approximation)
    + Gaussian RBFs (local flexibility).
    
    6 Chebyshev terms (T0..T5) on mapped domain t = x/3 in [-1,1]
    4 Gaussian RBFs centered at -2.25, -0.75, 0.75, 2.25 with moderate width
    
    This combines the stable global approximation of Chebyshev (current program)
    with localized bumps (from inspiration 2/3) to capture both smooth trends
    and local features of the target function.
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
    
    # Chebyshev polynomials T0..T5 via recurrence
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Gaussian RBFs with centers spread across the domain
    # Using 4 Gaussians at -2.25, -0.75, 0.75, 2.25
    # Width chosen to be moderate: sigma=1.0, so exp(-(x-c)^2/(2*sigma^2))
    sigma_sq = 2.0 * 1.0**2  # 2.0
    
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    diffs = x[None, :] - centers[:, None]  # shape (4, n)
    gaussians = np.exp(-diffs**2 / sigma_sq)  # shape (4, n)
    
    # Linear combination: 6 Chebyshev + 4 Gaussian = 10 terms
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * gaussians[0] +
         params[7] * gaussians[1] +
         params[8] * gaussians[2] +
         params[9] * gaussians[3])
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
