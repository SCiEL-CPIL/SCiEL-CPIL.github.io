"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with 10 fitted parameters.
    
    Features:
    0: constant
    1: t
    2: sin(t)
    3: cos(t)
    4: x
    5: x**3
    6: v
    7: v * |v|
    8: x * v
    9: sin(t) * x
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute features
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    f0 = np.ones_like(t)
    f1 = t
    f2 = sin_t
    f3 = cos_t
    f4 = x
    f5 = x ** 3
    f6 = v
    f7 = v * np.abs(v)
    f8 = x * v
    f9 = sin_t * x
    
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + 
            p4 * f4 + p5 * f5 + p6 * f6 + p7 * f7 + 
            p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
