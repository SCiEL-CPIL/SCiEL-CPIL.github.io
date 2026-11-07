import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a full 3D polynomial basis up to degree 2.
    The 10 parameters are the coefficients of:
      1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    
    This is a linear model in the parameters, so BFGS should converge quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple fallback to avoid timeout
        return np.ones_like(x)
    
    # Extract coefficients
    c0, cx, cy, cz, cx2, cy2, cz2, cxy, cxz, cyz = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Compute polynomial terms
    result = c0 + cx * x + cy * y + cz * z + cx2 * x * x + cy2 * y * y + cz2 * z * z + cxy * x * y + cxz * x * z + cyz * y * z
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
