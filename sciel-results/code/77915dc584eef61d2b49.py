"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: centered quadratic polynomial + 3 Gaussian RBFs.
    
    The quadratic part captures smooth low-order structure with good conditioning.
    The RBF part captures local/non-polynomial features that a global polynomial
    cannot fit well. All inputs are scaled to [-1, 1] for numerical stability.
    
    10 parameters total:
      p[0..6] : intercept, linear (3), squared (3) in centered/scaled vars
      p[7..9] : RBF amplitudes at 3 fixed centers
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # Scale inputs to [-1, 1]
    x_s = (x - 1.25) / 0.75
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    # Quadratic basis (7 terms)
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = y_s
    b3 = z_s
    b4 = x_s * x_s
    b5 = y_s * y_s
    b6 = z_s * z_s
    
    # RBF basis (3 terms) with fixed centers and width
    # Centers chosen to cover the domain: center, and two off-center points
    centers = np.array([
        [0.0, 0.0, 0.0],
        [-0.6, -0.6, -0.6],
        [0.6, 0.6, 0.6]
    ])
    width = 0.7
    
    rbf1 = np.exp(-((x_s - centers[0,0])**2 + (y_s - centers[0,1])**2 + (z_s - centers[0,2])**2) / (width**2))
    rbf2 = np.exp(-((x_s - centers[1,0])**2 + (y_s - centers[1,1])**2 + (z_s - centers[1,2])**2) / (width**2))
    rbf3 = np.exp(-((x_s - centers[2,0])**2 + (y_s - centers[2,1])**2 + (z_s - centers[2,2])**2) / (width**2))
    
    # Combine
    result = (p[0] * b0 +
              p[1] * b1 +
              p[2] * b2 +
              p[3] * b3 +
              p[4] * b4 +
              p[5] * b5 +
              p[6] * b6 +
              p[7] * rbf1 +
              p[8] * rbf2 +
              p[9] * rbf3)
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
