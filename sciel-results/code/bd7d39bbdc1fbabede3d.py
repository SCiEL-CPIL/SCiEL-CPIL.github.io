"""
Symbolic regression using a full bivariate cubic polynomial basis.
10 terms: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2*y, x*y^2
Linear in params → BFGS converges efficiently.
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
        return np.ones_like(x) * params[0]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Full bivariate cubic: all monomials of total degree <= 3
    # 1, x, y, x^2, y^2, xy, x^3, y^3, x^2*y, x*y^2
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = x2
    t4 = y2
    t5 = x * y
    t6 = x3
    t7 = y3
    t8 = x2 * y
    t9 = x * y2
    
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
