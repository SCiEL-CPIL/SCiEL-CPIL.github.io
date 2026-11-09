"""Rewritten program for symbolic regression task using rational Chebyshev model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a rational function model based on
    Chebyshev polynomials mapped to the domain [-2, 2]:
    
    y(x) = c + (a0 + a1*T1 + a2*T2 + a3*T3 + a4*T4) / (1 + b1*T1 + b2*T2 + b3*T3 + b4*T4)
    
    where T_i are Chebyshev polynomials of the first kind evaluated at t = x/2.
    The mapping t = x/2 ensures the basis functions vary over the entire input domain [-2, 2].
    
    The 10 params are: a0, a1, a2, a3, a4, b1, b2, b3, b4, and an offset c.
    """
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    a0, a1, a2, a3, a4, b1, b2, b3, b4, c = params
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T0 through T4 for t
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0
    
    # Numerator: a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4
    numerator = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: 1 + b1*T1 + b2*T2 + b3*T3 + b4*T4
    # The constant term is fixed at 1 to ensure denominator is non-zero near origin
    denominator = 1.0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Compute rational part with safe division
    # Use a large epsilon to avoid division by zero
    eps = 1e-8
    # Handle sign of denominator for safe division
    sign = np.sign(denominator)
    # If denominator is close to 0, replace with eps * sign (or eps if sign is 0)
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.where(sign == 0, 1.0, sign), denominator)
    rational_part = numerator / safe_denominator
    
    # Add offset
    result = c + rational_part
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
