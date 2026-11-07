"""Symbolic regression using a stable rational function with Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation using Chebyshev polynomials.
    y = P(t) / (1 + Q(t)^2)
    
    where t = x / 2 maps x in [-2, 2] to t in [-1, 1].
    P(t) and Q(t) are Chebyshev series of degree 4.
    
    params layout (length 10):
      params[0..4] = coefficients for P(t): p0, p1, p2, p3, p4
      params[5..9] = coefficients for Q(t): q0, q1, q2, q3, q4
    
    This model is structurally different from the sigmoid sum.
    It can model rational behaviors (poles, asymptotes) while the squared
    denominator ensures numerical stability (denominator >= 1).
    The Chebyshev basis provides better conditioning for BFGS than monomials.
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Map x to [-1, 1]
    t = x / 2.0
    
    # Extract parameters
    p_coeffs = params[0:5]
    q_coeffs = params[5:10]
    
    # Compute Chebyshev polynomials T0..T4
    # T0 = 1
    # T1 = t
    # T2 = 2*t^2 - 1
    # T3 = 4*t^3 - 3*t
    # T4 = 8*t^4 - 8*t^2 + 1
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t2 - 1.0
    T3 = 4.0 * t3 - 3.0 * t
    T4 = 8.0 * t4 - 8.0 * t2 + 1.0
    
    # Evaluate P(t)
    P = (p_coeffs[0] * T0 + 
         p_coeffs[1] * T1 + 
         p_coeffs[2] * T2 + 
         p_coeffs[3] * T3 + 
         p_coeffs[4] * T4)
         
    # Evaluate Q(t)
    Q = (q_coeffs[0] * T0 + 
         q_coeffs[1] * T1 + 
         q_coeffs[2] * T2 + 
         q_coeffs[3] * T3 + 
         q_coeffs[4] * T4)
         
    # Calculate result: y = P / (1 + Q^2)
    # Q^2 ensures denominator is >= 1, avoiding division by zero
    result = P / (1.0 + Q * Q)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
