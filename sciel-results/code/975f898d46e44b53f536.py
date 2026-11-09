# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a general polynomial basis.
    Replaces higher-order Duffing terms (x^4, v^4, x^2v^2) with lower-order
    terms (x^2, v^2, xv) which are likely more significant given the input ranges.
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    return (params[0] * x +
            params[1] * x2 +
            params[2] * x3 +
            params[3] * v +
            params[4] * v2 +
            params[5] * v3 +
            params[6] * x * v +
            params[7] * x * v2 +
            params[8] * x2 * v +
            params[9])
# EVOLVE-BLOCK-END
