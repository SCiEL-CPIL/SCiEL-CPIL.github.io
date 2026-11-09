"""
Hybrid basis for symbolic regression: orthogonal polynomial + localized RBF terms.
Linear in parameters so BFGS can fit efficiently. Combines global polynomial
representation with local Gaussian bumps for flexibility. Includes deadline guard.
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
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Orthogonal polynomial basis (Legendre) for global structure
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = (3t^2 - 1) / 2
    # P3(t) = (5t^3 - 3t) / 2
    
    P0 = np.ones_like(x)
    P1x = x
    P1y = y
    P2x = (3.0 * x2 - 1.0) * 0.5
    P2y = (3.0 * y2 - 1.0) * 0.5
    P3x = (5.0 * x3 - 3.0 * x) * 0.5
    P3y = (5.0 * y3 - 3.0 * y) * 0.5
    
    # Global polynomial terms (8 terms)
    t0 = P0          # Constant
    t1 = P1x         # Linear x
    t2 = P1y         # Linear y
    t3 = P2x         # Quadratic x (orthogonal)
    t4 = P2y         # Quadratic y (orthogonal)
    t5 = P1x * P1y   # Interaction xy
    t6 = P3x         # Cubic x (orthogonal)
    t7 = P3y         # Cubic y (orthogonal)
    
    # Localized RBF terms (2 terms) for local flexibility
    # These capture non-polynomial, localized behavior
    r2_center = x2 + y2
    t8 = np.exp(-2.0 * r2_center)       # Gaussian bump at origin
    t9 = np.exp(-1.0 * (x2 + y2 - 0.5)) # Broader Gaussian, slightly offset
    
    # Linear combination
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
