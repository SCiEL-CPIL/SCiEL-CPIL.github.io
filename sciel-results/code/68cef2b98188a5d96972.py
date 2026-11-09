# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with 10 fixed centers evenly spaced across [0.5, 2]^3
    # Linear in params (10 coefficients), so BFGS converges reliably.
    # Gaussian kernel: exp(-gamma * ||x - c_i||^2)
    
    n = len(x)
    
    # Define 10 fixed centers: a 2x2x2 grid at corners + 2 interior points
    # This gives good coverage of the domain
    cx = [0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 1.0, 1.5]
    cy = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 1.0, 1.5]
    cz = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.0, 1.5]
    
    # Fixed bandwidth gamma chosen for good local smoothness
    gamma = 2.0
    
    # Build RBF feature matrix: shape (n, 10)
    F = np.empty((n, 10), dtype=np.float64)
    for i in range(10):
        dx = x - cx[i]
        dy = y - cy[i]
        dz = z - cz[i]
        dist_sq = dx * dx + dy * dy + dz * dz
        F[:, i] = np.exp(-gamma * dist_sq)
    
    return F @ params
# EVOLVE-BLOCK-END
