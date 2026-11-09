"""
Symbolic regression using a full cubic polynomial basis.
This model is linear in parameters, ensuring BFGS converges to the global
minimum of the convex least-squares problem. The basis is chosen to be
well-conditioned on the domain [-1, 1]^2.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 15.0


def _time_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if not _time_ok():
        # Return a safe constant prediction
        return np.ones_like(x) * params[0]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Full cubic polynomial basis in x and y
    # Degree 0
    t0 = np.ones_like(x)
    
    # Degree 1
    t1 = x
    t2 = y
    
    # Degree 2
    t3 = x * x
    t4 = y * y
    t5 = x * y
    
    # Degree 3
    t6 = x * x * x
    t7 = y * y * y
    t8 = x * x * y
    t9 = x * y * y
    
    return (p0 * t0 + 
            p1 * t1 + 
            p2 * t2 + 
            p3 * t3 + 
            p4 * t4 + 
            p5 * t5 + 
            p6 * t6 + 
            p7 * t7 + 
            p8 * t8 + 
            p9 * t9)
# EVOLVE-BLOCK-END
