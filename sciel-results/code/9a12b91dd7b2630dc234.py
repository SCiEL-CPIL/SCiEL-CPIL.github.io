"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: fixed-center RBFs for local detail + low-order polynomial for global trend.
    
    The 10 parameters are:
    - params[0:6]: weights for 6 RBFs with fixed centers spread across [-2, 2]
    - params[6:10]: weights for 4 polynomial basis functions (1, x, x^2, x^3)
    
    Fixed RBF centers prevent BFGS from getting stuck in local minima or diverging,
    while the polynomial basis captures global smooth trends.
    """
    t = np.asarray(x, dtype=np.float64)
    
    # Fixed RBF centers spread across the domain [-2, 2]
    centers = np.array([-2.0, -1.3333, -0.6667, 0.0, 0.6667, 1.3333, 2.0])
    # Use 6 of the 7 centers to keep total params at 10
    centers = centers[:6]
    
    # RBF width (fixed, not a parameter to avoid optimization issues)
    sigma = 0.8
    
    # Compute RBF basis functions
    # t has shape (n,), centers has shape (6,)
    diff = t[:, np.newaxis] - centers[np.newaxis, :]  # (n, 6)
    rbf_basis = np.exp(-diff**2 / (2.0 * sigma**2))  # (n, 6)
    
    # Polynomial basis functions
    poly_basis = np.stack([
        np.ones_like(t),
        t,
        t**2,
        t**3
    ], axis=-1)  # (n, 4)
    
    # Combine and compute prediction
    result = rbf_basis @ params[0:6] + poly_basis @ params[6:10]
    
    return result
# EVOLVE-BLOCK-END
