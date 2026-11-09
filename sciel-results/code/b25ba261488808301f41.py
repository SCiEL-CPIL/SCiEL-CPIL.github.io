"""Symbolic regression with 3 inputs - richer basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 basis functions.
    
    Basis:
    p0: 1
    p1: x
    p2: y
    p3: z
    p4: x*y
    p5: x*z
    p6: y*z
    p7: x*y*z
    p8: 1/x
    p9: 1/y
    """
    f0 = 1.0
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    f8 = 1.0 / x
    f9 = 1.0 / y
    
    pred = (params[0] * f0 + params[1] * f1 + params[2] * f2 + params[3] * f3 +
            params[4] * f4 + params[5] * f5 + params[6] * f6 + params[7] * f7 +
            params[8] * f8 + params[9] * f9)
    return pred
# EVOLVE-BLOCK-END
