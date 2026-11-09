"""Rewritten program for symbolic regression task using rational Chebyshev model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a rational function model based on
    Chebyshev polynomials:
    
    y(x) = (a0 + a1*T1 + a2*T2 + a3*T3 + a4*T4) / (1 + b1*T1 + b2*T2 + b3*T3 + b4*T4)
    
    where T_i are Chebyshev polynomials of the first kind, and the denominator
    is constrained to be 1 + sum to ensure it doesn't vanish at x=0 when params
    are near zero, improving numerical stability.
    
    The 10 params are: a0, a1, a2, a3, a4, b1, b2, b3, b4, and an offset c.
    The final model is: c + rational_part(x)
    """
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    a0, a1, a2, a3, a4, b1, b2, b3, b4, c = params
    
    # Clip x to [-1, 1] for Chebyshev evaluation to avoid numerical issues
    # Chebyshev polynomials are defined on [-1, 1]
    xc = np.clip(x, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T0 through T4
    T0 = np.ones_like(xc)
    T1 = xc
    T2 = 2.0 * xc * xc - 1.0
    T3 = 4.0 * xc * xc * xc - 3.0 * xc
    T4 = 8.0 * xc * xc * xc * xc - 8.0 * xc * xc + 1.0
    
    # Numerator: a0 + a1*T1 + a2*T2 + a3*T3 + a4*T4
    numerator = a0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: 1 + b1*T1 + b2*T2 + b3*T3 + b4*T4
    # The constant term is fixed at 1 to ensure denominator is non-zero near origin
    denominator = 1.0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Compute rational part with safe division
    # Use a large epsilon to avoid division by zero
    eps = 1e-8
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-16), denominator)
    rational_part = numerator / safe_denominator
    
    # Add offset
    result = c + rational_part
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
