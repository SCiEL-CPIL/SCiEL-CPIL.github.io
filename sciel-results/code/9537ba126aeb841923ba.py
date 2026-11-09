"""Symbolic regression with 3 inputs using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(t: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_0(t), T_1(t), ..., T_max_degree(t).
    t should be in [-1, 1].
    Returns array of shape (len(t), max_degree + 1).
    """
    n = len(t)
    basis = np.zeros((n, max_degree + 1))
    basis[:, 0] = 1.0
    if max_degree >= 1:
        basis[:, 1] = t
        for k in range(2, max_degree + 1):
            basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Chebyshev polynomials mapped to [0.5, 2].
    
    We map each input variable from [0.5, 2] to [-1, 1] via:
        t = (2 * v - 2.5) / 1.5  =>  t = (4*v - 5) / 3
    
    Basis:
    - T_0(x) = 1
    - T_1(x) = t_x
    - T_2(x) = 2*t_x^2 - 1
    - T_0(y) = 1 (but we only use distinct terms)
    
    To use exactly 10 parameters, we construct a basis:
    1. 1 (constant)
    2. T_1(x)
    3. T_2(x)
    4. T_1(y)
    5. T_2(y)
    6. T_1(z)
    7. T_2(z)
    8. T_1(x) * T_1(y)
    9. T_1(x) * T_1(z)
    10. T_1(y) * T_1(z)
    
    This gives 10 basis functions.
    """
    p = params
    
    # Map inputs from [0.5, 2] to [-1, 1]
    tx = (4.0 * x - 5.0) / 3.0
    ty = (4.0 * y - 5.0) / 3.0
    tz = (4.0 * z - 5.0) / 3.0
    
    # Compute Chebyshev polynomials
    T1x = tx
    T2x = 2.0 * tx * tx - 1.0
    
    T1y = ty
    T2y = 2.0 * ty * ty - 1.0
    
    T1z = tz
    T2z = 2.0 * tz * tz - 1.0
    
    # Interaction terms
    T1x_T1y = T1x * T1y
    T1x_T1z = T1x * T1z
    T1y_T1z = T1y * T1z
    
    # Combine with parameters
    result = (
        p[0] * 1.0 +
        p[1] * T1x +
        p[2] * T2x +
        p[3] * T1y +
        p[4] * T2y +
        p[5] * T1z +
        p[6] * T2z +
        p[7] * T1x_T1y +
        p[8] * T1x_T1z +
        p[9] * T1y_T1z
    )
    
    return result
# EVOLVE-BLOCK-END
