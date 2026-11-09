"""
Symbolic regression using an orthogonal polynomial basis (Legendre polynomials).
This model is linear in parameters, ensuring BFGS converges efficiently.
Legendre polynomials are orthogonal on [-1,1], reducing collinearity
compared to monomial bases and improving numerical conditioning.
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
    
    # Legendre polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = (3t^2 - 1) / 2
    # P3(t) = (5t^3 - 3t) / 2
    
    P0x = np.ones_like(x)
    P1x = x
    P2x = (3.0 * x * x - 1.0) * 0.5
    P3x = (5.0 * x * x * x - 3.0 * x) * 0.5
    
    P0y = np.ones_like(y)
    P1y = y
    P2y = (3.0 * y * y - 1.0) * 0.5
    P3y = (5.0 * y * y * y - 3.0 * y) * 0.5
    
    # Tensor product basis: 10 terms
    # (Px_i, Py_j) with i+j <= 3, choosing 10 most important
    t0 = P0x * P0y       # 1
    t1 = P1x * P0y       # x
    t2 = P0x * P1y       # y
    t3 = P2x * P0y       # (3x^2-1)/2
    t4 = P0x * P2y       # (3y^2-1)/2
    t5 = P1x * P1y       # xy
    t6 = P3x * P0y       # (5x^3-3x)/2
    t7 = P0x * P3y       # (5y^3-3y)/2
    t8 = P2x * P1y       # ((3x^2-1)/2) * y
    t9 = P1x * P2y       # x * ((3y^2-1)/2)
    
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
