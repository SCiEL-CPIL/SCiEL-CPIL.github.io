"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a cubic polynomial basis in 2D.
    Basis: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    This uses exactly 10 parameters and is linear in the parameters,
    allowing BFGS to reliably find the least-squares solution.
    """
    p = params
    
    # Pre-compute powers for efficiency and clarity
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Construct the prediction
    # p[0]: constant
    # p[1]: x
    # p[2]: y
    # p[3]: x^2
    # p[4]: xy
    # p[5]: y^2
    # p[6]: x^3
    # p[7]: x^2y
    # p[8]: xy^2
    # p[9]: y^3
    
    pred = (p[0] * np.ones_like(x) +
            p[1] * x +
            p[2] * y +
            p[3] * x2 +
            p[4] * xy +
            p[5] * y2 +
            p[6] * x3 +
            p[7] * x2y +
            p[8] * xy2 +
            p[9] * y3)
            
    return pred
# EVOLVE-BLOCK-END
