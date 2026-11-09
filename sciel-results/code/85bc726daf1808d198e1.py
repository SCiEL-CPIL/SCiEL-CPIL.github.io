"""Symbolic regression with Chebyshev separable product basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_values(t, max_degree):
    """Compute Chebyshev polynomials T_0(t), T_1(t), ..., T_max_degree(t) for array t."""
    t = np.asarray(t, dtype=np.float64)
    n = t.shape[0]
    if max_degree == 0:
        return np.ones((n, 1))
    if max_degree == 1:
        return np.column_stack([np.ones(n), t])
    
    # T_0 = 1, T_1 = t, T_{k+1} = 2t*T_k - T_{k-1}
    T = np.zeros((n, max_degree + 1))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, max_degree + 1):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis: Chebyshev polynomials in x (degrees 0-3) 
    and Chebyshev polynomials in y (degrees 0-2).
    
    Total basis functions: 4 * 3 = 12, but we only have 10 params.
    We use a subset: T_0(x)*T_0(y), T_1(x)*T_0(y), T_2(x)*T_0(y), T_3(x)*T_0(y),
                     T_0(x)*T_1(y), T_1(x)*T_1(y), T_2(x)*T_1(y), T_3(x)*T_1(y),
                     T_0(x)*T_2(y), T_1(x)*T_2(y)
    That's 10 terms.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.shape[0]
    
    # Compute Chebyshev values
    Tx = _chebyshev_values(x, 3)  # shape (n, 4): T_0, T_1, T_2, T_3
    Ty = _chebyshev_values(y, 2)  # shape (n, 3): T_0, T_1, T_2
    
    p = params[:10]
    
    # Build the 10 basis terms:
    # Term 0: T_0(x) * T_0(y) = 1
    # Term 1: T_1(x) * T_0(y) = x
    # Term 2: T_2(x) * T_0(y) = 2x^2 - 1
    # Term 3: T_3(x) * T_0(y) = 4x^3 - 3x
    # Term 4: T_0(x) * T_1(y) = y
    # Term 5: T_1(x) * T_1(y) = x * y
    # Term 6: T_2(x) * T_1(y) = (2x^2 - 1) * y
    # Term 7: T_3(x) * T_1(y) = (4x^3 - 3x) * y
    # Term 8: T_0(x) * T_2(y) = 2y^2 - 1
    # Term 9: T_1(x) * T_2(y) = x * (2y^2 - 1)
    
    result = (
        p[0] * Tx[:, 0] * Ty[:, 0] +
        p[1] * Tx[:, 1] * Ty[:, 0] +
        p[2] * Tx[:, 2] * Ty[:, 0] +
        p[3] * Tx[:, 3] * Ty[:, 0] +
        p[4] * Tx[:, 0] * Ty[:, 1] +
        p[5] * Tx[:, 1] * Ty[:, 1] +
        p[6] * Tx[:, 2] * Ty[:, 1] +
        p[7] * Tx[:, 3] * Ty[:, 1] +
        p[8] * Tx[:, 0] * Ty[:, 2] +
        p[9] * Tx[:, 1] * Ty[:, 2]
    )
    
    return result
# EVOLVE-BLOCK-END
