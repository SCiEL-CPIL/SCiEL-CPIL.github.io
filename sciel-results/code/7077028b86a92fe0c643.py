"""Symbolic regression with structured basis for common functional forms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structured basis designed to capture common symbolic regression targets.
    
    Uses 10 parameters arranged as:
    - params[0]: constant
    - params[1]: x
    - params[2]: y
    - params[3]: z
    - params[4]: x*y
    - params[5]: x*z
    - params[6]: y*z
    - params[7]: x*y*z
    - params[8]: x^2
    - params[9]: y^2
    """
    p = params
    
    # Basic terms
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Interaction and nonlinear terms
    xy = x * y
    xz = x * z
    yz = y * z
    xyz = x * y * z
    x2 = x * x
    y2 = y * y
    
    result = (p[0]
              + p[1] * x
              + p[2] * y
              + p[3] * z
              + p[4] * xy
              + p[5] * xz
              + p[6] * yz
              + p[7] * xyz
              + p[8] * x2
              + p[9] * y2)
    
    return result
# EVOLVE-BLOCK-END
