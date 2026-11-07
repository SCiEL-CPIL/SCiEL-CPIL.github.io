"""Symbolic regression candidate: rich polynomial + RBF basis with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining:
    - Full quadratic polynomial (10 terms: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz)
    - Plus additional cubic cross terms and two 3D Gaussian RBFs for local flexibility.
    
    The 10 parameters are structured as:
    0: bias
    1: weight for u_x
    2: weight for u_y
    3: weight for u_z
    4: weight for u_x^2 + u_y^2 + u_z^2
    5: weight for u_x*u_y + u_x*u_z + u_y*u_z
    6: weight for RBF 1 (center at origin)
    7: log-width for RBF 1
    8: weight for RBF 2 (center at (0.5, -0.5, 0.5))
    9: log-width for RBF 2
    
    Normalization: u = (v - 1.25) / 0.75 maps [0.5, 2] to [-1, 1]
    """
    p = params
    
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Normalize inputs to [-1, 1]
    ux = (x - 1.25) / 0.75
    uy = (y - 1.25) / 0.75
    uz = (z - 1.25) / 0.75
    
    # Quadratic terms
    quad_iso = ux * ux + uy * uy + uz * uz
    cross_iso = ux * uy + ux * uz + uy * uz
    
    # RBF 1: center at (0, 0, 0)
    width1 = np.exp(p[7])
    dist1_sq = ux * ux + uy * uy + uz * uz
    rbf1 = np.exp(-dist1_sq / (2.0 * width1 * width1))
    
    # RBF 2: center at (0.5, -0.5, 0.5)
    width2 = np.exp(p[9])
    dx2 = ux - 0.5
    dy2 = uy + 0.5
    dz2 = uz - 0.5
    dist2_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    rbf2 = np.exp(-dist2_sq / (2.0 * width2 * width2))
    
    # Combine all terms
    result = (p[0] +
              p[1] * ux +
              p[2] * uy +
              p[3] * uz +
              p[4] * quad_iso +
              p[5] * cross_iso +
              p[6] * rbf1 +
              p[8] * rbf2)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
