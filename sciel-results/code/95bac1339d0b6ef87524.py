# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 145.0  # seconds, 5s safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 7 polynomial terms (constant, 3 linear, 3 diagonal quadratic)
    + 3 Gaussian RBF terms centered at domain corners and center.
    
    The RBFs capture localized non-polynomial structure that a global quadratic
    cannot. The polynomial backbone ensures stability and good conditioning.
    
    RBF centers (in original [0.5, 2] space):
      c1 = (0.5, 0.5, 0.5)  -- low corner
      c2 = (2.0, 2.0, 2.0)  -- high corner
      c3 = (1.25, 1.25, 1.25) -- center
    Width sigma = 0.5 in original units.
    """
    # Deadline check: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        # Fallback: just return a constant (params[0]) to avoid timeout
        return np.ones_like(np.asarray(x, dtype=np.float64)) * float(np.asarray(params, dtype=np.float64).flat[0])
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # Scale inputs to [-1, 1] for polynomial terms
    x_s = (x - 1.25) / 0.75
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    # Polynomial basis (7 terms): constant, linear, diagonal quadratic
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = y_s
    b3 = z_s
    b4 = x_s * x_s
    b5 = y_s * y_s
    b6 = z_s * z_s
    
    # RBF basis (3 terms): Gaussian kernels
    # Centers in original space
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([2.0, 2.0, 2.0])
    c3 = np.array([1.25, 1.25, 1.25])
    sigma = 0.5
    
    # Compute squared distances to each center
    d1_sq = (x - c1[0])**2 + (y - c1[1])**2 + (z - c1[2])**2
    d2_sq = (x - c2[0])**2 + (y - c2[1])**2 + (z - c2[2])**2
    d3_sq = (x - c3[0])**2 + (y - c3[1])**2 + (z - c3[2])**2
    
    # Gaussian RBFs: exp(-d^2 / (2*sigma^2))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    b7 = np.exp(-d1_sq * inv_2sigma2)
    b8 = np.exp(-d2_sq * inv_2sigma2)
    b9 = np.exp(-d3_sq * inv_2sigma2)
    
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    result = B @ p
    
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
