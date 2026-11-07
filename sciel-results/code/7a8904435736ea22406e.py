"""Symbolic regression with 1 input using RBF basis for flexible non-linear fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Gaussian RBF basis with 10 centers evenly spaced in [0.1, 6].
    The model is linear in params: pred = sum_j params[j] * phi_j(x).
    
    Key design choices:
    1. RBF centers at 10 evenly spaced points in [0.1, 6].
    2. Fixed bandwidth sigma chosen to balance local and global fitting.
    3. Linear model in params so BFGS converges to global least-squares optimum.
    4. This allows fitting arbitrary smooth non-linear shapes without polynomial conditioning issues.
    
    The RBF basis is more flexible than polynomials for capturing non-polynomial structure
    and avoids the high condition number issues of high-degree polynomial bases.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([], dtype=np.float64)
    
    # 10 RBF centers evenly spaced in [0.1, 6]
    x_min, x_max = 0.1, 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed bandwidth: sigma = (x_max - x_min) / (2 * sqrt(10)) ~ 0.94
    # This gives reasonable overlap between adjacent RBFs
    sigma = (x_max - x_min) / (2.0 * np.sqrt(10.0))
    
    # Compute RBF values: phi_j(x) = exp(-((x - c_j)^2) / (2 * sigma^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination
    return rbf @ params
# EVOLVE-BLOCK-END
