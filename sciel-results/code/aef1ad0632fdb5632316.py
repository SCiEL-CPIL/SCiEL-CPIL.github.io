"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete quadratic polynomial basis.
    Inputs x, y, z are in [0.5, 2].
    
    Basis functions:
    1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    
    This basis spans all polynomials of degree <= 2, providing a robust
    approximation for smooth functions in the given domain. It avoids the
    potential bias of missing pure quadratic terms (x^2, y^2, z^2) present
    in the previous version.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis functions (10 terms) - Complete quadratic polynomial
    
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * x
    f5 = y * y
    f6 = z * z
    f7 = x * y
    f8 = x * z
    f9 = y * z
    
    result = (params[0] * f0 +
              params[1] * f1 +
              params[2] * f2 +
              params[3] * f3 +
              params[4] * f4 +
              params[5] * f5 +
              params[6] * f6 +
              params[7] * f7 +
              params[8] * f8 +
              params[9] * f9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
