"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # 10 basis functions: linear in params for fast BFGS convergence
    # Using a rich set of basis functions to cover polynomials and trigonometric terms
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    p4 = x**4
    p5 = np.sin(x)
    p6 = np.cos(x)
    p7 = np.sin(2.0 * x)
    p8 = np.cos(2.0 * x)
    p9 = x**5
    
    return (params[0] * p0 + params[1] * p1 + params[2] * p2 +
            params[3] * p3 + params[4] * p4 + params[5] * p5 +
            params[6] * p6 + params[7] * p7 + params[8] * p8 +
            params[9] * p9)
# EVOLVE-BLOCK-END
