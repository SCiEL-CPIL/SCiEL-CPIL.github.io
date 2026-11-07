# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Hybrid model combining local Gaussian RBFs with a global polynomial basis.
    - 6 parameters for a quadratic polynomial basis (1, x, y, x^2, xy, y^2)
    - 4 parameters for 4 Gaussian RBF centers covering the domain
    
    This combines the global trend capture of polynomials with local
    non-linear adjustments from RBFs. The RBF centers are fixed at
    the corners of the domain to capture boundary effects, while the
    polynomial basis handles the global structure.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Global polynomial basis (6 features)
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = x2
    f4 = xy
    f5 = y2
    
    # Local RBF component (4 centers at corners)
    centers = np.array([
        [0.7, 0.7],
        [-0.7, 0.7],
        [0.7, -0.7],
        [-0.7, -0.7]
    ])
    
    sigma = 0.8
    sigma2 = 2 * sigma * sigma
    
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    
    dist_sq = dx * dx + dy * dy
    gaussians = np.exp(-dist_sq / sigma2)  # shape (N, 4)
    
    # Combine: polynomial part + RBF part
    poly_part = (p[0] * f0 + p[1] * f1 + p[2] * f2 +
                 p[3] * f3 + p[4] * f4 + p[5] * f5)
    
    rbf_part = gaussians @ p[6:10]
    
    return poly_part + rbf_part
# EVOLVE-BLOCK-END
