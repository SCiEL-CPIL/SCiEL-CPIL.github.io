"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with fixed orthogonal frequencies up to 5th harmonic.
    
    Basis:
    0: constant (1)
    1: sin(x)
    2: cos(x)
    3: sin(2x)
    4: cos(2x)
    5: sin(3x)
    6: cos(3x)
    7: sin(4x)
    8: cos(4x)
    9: sin(5x)
    
    This captures periodic/oscillatory structures with well-conditioned
    gradients for BFGS optimization.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Precompute trig terms
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    sin2 = np.sin(2.0 * x)
    cos2 = np.cos(2.0 * x)
    sin3 = np.sin(3.0 * x)
    cos3 = np.cos(3.0 * x)
    sin4 = np.sin(4.0 * x)
    cos4 = np.cos(4.0 * x)
    sin5 = np.sin(5.0 * x)
    
    # Linear combination of basis functions
    result = (p[0] +
              p[1] * sin1 + p[2] * cos1 +
              p[3] * sin2 + p[4] * cos2 +
              p[5] * sin3 + p[6] * cos3 +
              p[7] * sin4 + p[8] * cos4 +
              p[9] * sin5)
    
    return result
# EVOLVE-BLOCK-END
