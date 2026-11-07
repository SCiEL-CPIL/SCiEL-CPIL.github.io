"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Precompute basis functions
    x2 = x * x
    x3 = x2 * x
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    tanh_x = np.tanh(x)
    sech_x = 1.0 / np.cosh(x)
    
    # 10-parameter linear combination of basis functions
    # params: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # Basis:  [1,  x,  x^2, x^3, sin(x), cos(x), exp(x), exp(-x), tanh(x), sech(x)]
    
    result = (
        params[0] * 1.0 +
        params[1] * x +
        params[2] * x2 +
        params[3] * x3 +
        params[4] * sin_x +
        params[5] * cos_x +
        params[6] * exp_x +
        params[7] * exp_neg_x +
        params[8] * tanh_x +
        params[9] * sech_x
    )
    
    return result
# EVOLVE-BLOCK-END
