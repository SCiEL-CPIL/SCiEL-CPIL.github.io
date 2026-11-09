"""Symbolic regression with mixed polynomial and reciprocal basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed basis including linear, quadratic, and reciprocal terms.
    
    Uses 10 parameters arranged as:
    - params[0]: constant
    - params[1]: x
    - params[2]: y
    - params[3]: z
    - params[4]: x^2
    - params[5]: y^2
    - params[6]: z^2
    - params[7]: 1/x
    - params[8]: 1/y
    - params[9]: 1/z
    """
    p = params
    
    # Ensure inputs are positive (given [0.5, 2])
    # Using raw values for polynomial terms and reciprocals
    
    result = (p[0] * 1.0
              + p[1] * x
              + p[2] * y
              + p[3] * z
              + p[4] * (x * x)
              + p[5] * (y * y)
              + p[6] * (z * z)
              + p[7] * (1.0 / x)
              + p[8] * (1.0 / y)
              + p[9] * (1.0 / z))
    
    return result
# EVOLVE-BLOCK-END
