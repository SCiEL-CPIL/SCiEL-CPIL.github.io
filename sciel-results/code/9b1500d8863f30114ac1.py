"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with fixed non-integer frequencies and cross-term interactions.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: sin(2.0*x)
    2: cos(2.0*x)
    3: sin(3.0*y)
    4: cos(3.0*y)
    5: sin(4.0*z)
    6: cos(4.0*z)
    7: sin(x + y)
    8: sin(y + z)
    9: sin(x + z)
    
    This representation captures periodic/oscillatory dependencies and
    non-linear couplings between variables.
    """
    p = params.flatten()
    
    # Ensure we have at least 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a4 = p[4]
    a5 = p[5]
    a6 = p[6]
    a7 = p[7]
    a8 = p[8]
    a9 = p[9]
    
    # Precompute trigonometric terms
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    sin_3y = np.sin(3.0 * y)
    cos_3y = np.cos(3.0 * y)
    sin_4z = np.sin(4.0 * z)
    cos_4z = np.cos(4.0 * z)
    sin_xy = np.sin(x + y)
    sin_yz = np.sin(y + z)
    sin_xz = np.sin(x + z)
    
    # Build the expression
    result = a0
    result += a1 * sin_2x
    result += a2 * cos_2x
    result += a3 * sin_3y
    result += a4 * cos_3y
    result += a5 * sin_4z
    result += a6 * cos_4z
    result += a7 * sin_xy
    result += a8 * sin_yz
    result += a9 * sin_xz
    
    return result
# EVOLVE-BLOCK-END
