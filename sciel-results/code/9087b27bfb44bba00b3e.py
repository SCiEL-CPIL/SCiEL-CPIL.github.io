"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier Series approach
    # Map x in [-3, 3] to u in [-pi, pi]
    u = x * (np.pi / 3.0)
    
    # Basis functions:
    # p0: 1
    # p1: cos(u)
    # p2: sin(u)
    # p3: cos(2u)
    # p4: sin(2u)
    # p5: cos(3u)
    # p6: sin(3u)
    # p7: cos(4u)
    # p8: sin(4u)
    # p9: cos(5u)
    
    result = params[0] * np.ones_like(x)
    
    result += params[1] * np.cos(u)
    result += params[2] * np.sin(u)
    result += params[3] * np.cos(2 * u)
    result += params[4] * np.sin(2 * u)
    result += params[5] * np.cos(3 * u)
    result += params[6] * np.sin(3 * u)
    result += params[7] * np.cos(4 * u)
    result += params[8] * np.sin(4 * u)
    result += params[9] * np.cos(5 * u)
    
    return result
# EVOLVE-BLOCK-END
