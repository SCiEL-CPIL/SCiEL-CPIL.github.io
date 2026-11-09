"""Rewritten program for symbolic regression task using rational Chebyshev model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_matrix(t: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Compute Chebyshev polynomials T_0(t), T_1(t), ..., T_{order-1}(t) for array t.
    Returns shape (len(t), order).
    """
    n = len(t)
    T = np.zeros((n, order), dtype=np.float64)
    T[:, 0] = 1.0
    if order > 1:
        T[:, 1] = t
    for k in range(2, order):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    return T


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Chebyshev model:
    y(x) = (sum_{i=0}^{4} a_i * T_i(t)) / (1 + sum_{j=1}^{4} b_j * T_j(t))
    
    where t = x/2 maps [-2,2] -> [-1,1], T_i are Chebyshev polynomials of first kind.
    
    params[0:5]  = numerator coefficients a_0..a_4
    params[5:10] = denominator coefficients b_1..b_4 (b_0 fixed to 1)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev evaluation
    t = 0.5 * x
    
    # Compute Chebyshev basis up to order 4 (5 terms: T0..T4)
    T = _chebyshev_matrix(t, order=5)
    
    # Numerator: a_0*T0 + a_1*T1 + a_2*T2 + a_3*T3 + a_4*T4
    a = params[0:5]
    numerator = T @ a
    
    # Denominator: 1 + b_1*T1 + b_2*T2 + b_3*T3 + b_4*T4
    b = params[5:10]
    denominator = 1.0 + T[:, 1] * b[0] + T[:, 2] * b[1] + T[:, 3] * b[2] + T[:, 4] * b[3]
    
    # Avoid division by zero or near-zero
    # Use a small epsilon to keep denominator away from zero
    eps = 1e-8
    sign = np.sign(denominator)
    sign = np.where(sign == 0, 1.0, sign)
    denominator_safe = sign * np.maximum(np.abs(denominator), eps)
    
    result = numerator / denominator_safe
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
