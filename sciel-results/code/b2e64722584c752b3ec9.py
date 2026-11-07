"""Symbolic regression with 2 inputs using a rich basis and robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_t(n, x):
    """Compute Chebyshev polynomial T_n(x) for n=0,1,2,3,4."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    elif n == 2:
        return 2 * x**2 - 1
    elif n == 3:
        return 4 * x**3 - 3 * x
    elif n == 4:
        return 8 * x**4 - 8 * x**2 + 1
    else:
        raise ValueError("n must be 0-4")


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function f(x,y) using a tensor-product Chebyshev basis.
    
    The basis uses T_i(x) * T_j(y) for i,j in {0,1,2,3,4}, giving 25 terms.
    We use 10 params: the first 10 basis functions in a structured order
    that captures low-degree terms first.
    
    Basis order (10 terms):
    0: T_0(x)*T_0(y) = 1
    1: T_1(x)*T_0(y) = x
    2: T_0(x)*T_1(y) = y
    3: T_2(x)*T_0(y) = 2x^2 - 1
    4: T_1(x)*T_1(y) = xy
    5: T_0(x)*T_2(y) = 2y^2 - 1
    6: T_3(x)*T_0(y) = 4x^3 - 3x
    7: T_2(x)*T_1(y) = (2x^2-1)y
    8: T_1(x)*T_2(y) = x(2y^2-1)
    9: T_0(x)*T_3(y) = 4y^3 - 3y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Compute Chebyshev polynomials up to degree 3
    T0x = np.ones_like(x)
    T1x = x.copy()
    T2x = 2 * x**2 - 1
    T3x = 4 * x**3 - 3 * x
    
    T0y = np.ones_like(y)
    T1y = y.copy()
    T2y = 2 * y**2 - 1
    T3y = 4 * y**3 - 3 * y
    
    # Build the 10 basis functions
    basis = np.stack([
        T0x * T0y,   # 0: 1
        T1x * T0y,   # 1: x
        T0x * T1y,   # 2: y
        T2x * T0y,   # 3: 2x^2 - 1
        T1x * T1y,   # 4: xy
        T0x * T2y,   # 5: 2y^2 - 1
        T3x * T0y,   # 6: 4x^3 - 3x
        T2x * T1y,   # 7: (2x^2-1)y
        T1x * T2y,   # 8: x(2y^2-1)
        T0x * T3y,   # 9: 4y^3 - 3y
    ], axis=-1)
    
    # Compute prediction as dot product with params
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
