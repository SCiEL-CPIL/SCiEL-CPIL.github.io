"""Symbolic regression with 2 inputs; EVOLVE-BLOCK contains the full solution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression that recovers an unknown function f(x, y).
    The evaluator fits params (length 10) via BFGS starting from ones.
    We design a rich basis such that a linear combination can approximate
    many smooth functions on [-1,1]^2.
    
    Basis functions (10 total):
    1. 1 (constant)
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. x^2*y
    8. x*y^2
    9. x^2*y^2
    10. x^3*y + x*y^3  (symmetric cubic cross term)
    
    This spans a rich polynomial space up to degree 4, which should
    approximate most smooth functions well on the unit square.
    """
    # Ensure inputs are proper arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis matrix: shape (n_points, 10)
    # p[0]: constant
    # p[1]: x
    # p[2]: y
    # p[3]: x^2
    # p[4]: y^2
    # p[5]: x*y
    # p[6]: x^2*y
    # p[7]: x*y^2
    # p[8]: x^2*y^2
    # p[9]: x^3*y + x*y^3
    
    p = params
    
    pred = (p[0] * np.ones_like(x) +
            p[1] * x +
            p[2] * y +
            p[3] * x**2 +
            p[4] * y**2 +
            p[5] * x * y +
            p[6] * x**2 * y +
            p[7] * x * y**2 +
            p[8] * x**2 * y**2 +
            p[9] * (x**3 * y + x * y**3))
    
    return pred
# EVOLVE-BLOCK-END
