"""Symbolic regression with 3 inputs using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_poly(t, n):
    """Compute Chebyshev polynomial of the first kind of degree n at points t."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t.copy()
    T_nm2 = np.ones_like(t)
    T_nm1 = t.copy()
    for k in range(2, n + 1):
        T_n = 2.0 * t * T_nm1 - T_nm2
        T_nm2 = T_nm1
        T_nm1 = T_n
    return T_nm1


def _map_to_unit(u):
    """Map from [0.5, 2] to [-1, 1]."""
    return (2.0 * u - (0.5 + 2.0)) / (2.0 - 0.5)


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Chebyshev polynomial basis.
    
    We use a total of 10 parameters arranged as:
    - params[0]: constant term
    - params[1:4]: linear Chebyshev terms T1(x), T1(y), T1(z)
    - params[4:7]: quadratic Chebyshev terms T2(x), T2(y), T2(z)
    - params[7:10]: interaction terms T1(x)*T1(y), T1(x)*T1(z), T1(y)*T1(z)
    
    The inputs are mapped from [0.5, 2] to [-1, 1] for the Chebyshev polynomials.
    """
    p = params
    
    # Map inputs to [-1, 1]
    tx = _map_to_unit(x)
    ty = _map_to_unit(y)
    tz = _map_to_unit(z)
    
    # Compute Chebyshev polynomials
    T1x = tx
    T1y = ty
    T1z = tz
    T2x = 2.0 * tx * tx - 1.0
    T2y = 2.0 * ty * ty - 1.0
    T2z = 2.0 * tz * tz - 1.0
    
    # Combine basis functions with parameters
    result = (
        p[0]
        + p[1] * T1x
        + p[2] * T1y
        + p[3] * T1z
        + p[4] * T2x
        + p[5] * T2y
        + p[6] * T2z
        + p[7] * T1x * T1y
        + p[8] * T1x * T1z
        + p[9] * T1y * T1z
    )
    
    return result
# EVOLVE-BLOCK-END
