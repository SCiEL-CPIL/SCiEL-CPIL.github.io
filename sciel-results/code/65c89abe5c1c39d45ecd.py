"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with fixed non-integer frequencies and cross-term interactions.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: sin(1.5*x)
    2: cos(1.5*x)
    3: sin(2.5*y)
    4: cos(2.5*y)
    5: sin(3.5*z)
    6: cos(3.5*z)
    7: sin(1.2*(x+y))
    8: sin(1.2*(y+z))
    9: sin(1.2*(x+z))
    
    Non-integer frequencies avoid aliasing with the uniform sampling range [0.5, 2].
    Cross-terms capture interaction effects between input variables.
    """
    p = params.flatten()
    
    # Ensure we have exactly 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    elif len(p) > 10:
        p = p[:10]
    
    # Build the expression using precomputed basis functions
    s15x = np.sin(1.5 * x)
    c15x = np.cos(1.5 * x)
    s25y = np.sin(2.5 * y)
    c25y = np.cos(2.5 * y)
    s35z = np.sin(3.5 * z)
    c35z = np.cos(3.5 * z)
    s12xy = np.sin(1.2 * (x + y))
    s12yz = np.sin(1.2 * (y + z))
    s12xz = np.sin(1.2 * (x + z))
    
    result = (p[0] * 1.0 +
              p[1] * s15x +
              p[2] * c15x +
              p[3] * s25y +
              p[4] * c25y +
              p[5] * s35z +
              p[6] * c35z +
              p[7] * s12xy +
              p[8] * s12yz +
              p[9] * s12xz)
    
    return result
# EVOLVE-BLOCK-END
