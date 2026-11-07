"""Symbolic regression candidate: hybrid RBF and polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining normalized quadratic polynomial terms with
    two 3D Gaussian Radial Basis Functions (RBFs).
    
    The 10 parameters are structured as:
    0: bias (intercept)
    1: u_x
    2: u_y
    3: u_z
    4: u_x^2 + u_y^2 + u_z^2  (isotropic quadratic)
    5: u_x*u_y + u_x*u_z + u_y*u_z (isotropic cross)
    6: weight for RBF 1
    7: width for RBF 1 (log-scale for positivity)
    8: weight for RBF 2
    9: width for RBF 2 (log-scale for positivity)
    
    RBF centers are fixed at (0, 0, 0) and (0.5, -0.5, 0.5) in normalized space.
    This provides local flexibility to capture non-polynomial features like
    exponentials or logs, while the polynomial terms handle global trends.
    
    Normalization maps [0.5, 2] to [-1, 1] via:
        u = (v - 1.25) / 0.75
    """
    p = params
    
    # Deadline guard: track elapsed time and return early if needed
    t0 = time.time()
    deadline = t0 + 115.0  # 5s safety margin before 120s limit
    
    # Normalize inputs to [-1, 1]
    ux = (x - 1.25) / 0.75
    uy = (y - 1.25) / 0.75
    uz = (z - 1.25) / 0.75
    
    # Isotropic quadratic and cross terms
    quad_iso = ux * ux + uy * uy + uz * uz
    cross_iso = ux * uy + ux * uz + uy * uz
    
    # RBF 1: center at (0, 0, 0)
    # Width in log-space to ensure positivity and smooth gradient
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
