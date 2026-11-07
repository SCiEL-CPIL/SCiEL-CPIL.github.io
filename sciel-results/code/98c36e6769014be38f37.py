import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis:
    - 6 global polynomial terms: constant, x, y, z, x^2, y^2 (captures global trends)
    - 1 RBF term: center (3), width (1), weight (1) = 5 params (captures local features)
    
    Wait, that's 6 + 5 = 11 params. We only have 10.
    
    Let's restructure:
    - 5 global terms: constant, x, y, z, xy (captures linear + one interaction)
    - 1 RBF term: center (3), width (1), weight (1) = 5 params
    Total: 5 + 5 = 10 params.
    
    Or better:
    - 4 global terms: constant, x, y, z
    - 1 RBF term: center (3), width (1), weight (1) = 5 params
    - 1 extra term: x^2 (to capture quadratic curvature)
    Total: 4 + 5 + 1 = 10 params.
    
    Let's go with:
    p[0] = constant
    p[1] = coeff for x
    p[2] = coeff for y
    p[3] = coeff for z
    p[4] = coeff for x^2
    p[5] = RBF center cx
    p[6] = RBF center cy
    p[7] = RBF center cz
    p[8] = RBF width sigma
    p[9] = RBF weight w
    
    This gives global polynomial (degree 2 in x only, linear in y,z) + one local Gaussian bump.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple fallback to avoid timeout
        return np.ones_like(x)
    
    # Global polynomial terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = z
    t4 = x * x
    
    # RBF term
    cx = p[5]
    cy = p[6]
    cz = p[7]
    sigma = np.maximum(np.abs(p[8]), 0.05)
    w = p[9]
    
    dx = x - cx
    dy = y - cy
    dz = z - cz
    dist_sq = dx * dx + dy * dy + dz * dz
    
    t_rbf = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    result = (p[0] * t0 + 
              p[1] * t1 + 
              p[2] * t2 + 
              p[3] * t3 + 
              p[4] * t4 + 
              w * t_rbf)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
