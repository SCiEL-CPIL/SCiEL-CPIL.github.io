"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete polynomial basis up to degree 4
    in two variables, with 10 parameters.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: x*y
    5: y^2
    6: x^3
    7: x^2*y
    8: x*y^2
    9: y^3
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    
    return p0 + p1 * x + p2 * y + p3 * x2 + p4 * xy + p5 * y2 + p6 * x3 + p7 * x2y + p8 * xy2 + p9 * y3
# EVOLVE-BLOCK-END
