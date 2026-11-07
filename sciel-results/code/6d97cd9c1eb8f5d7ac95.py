"""Symbolic regression using rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = N(x) / D(x)
    
    where N(x) is a degree-3 polynomial and D(x) is a degree-2 polynomial + 1.
    
    N(x) = p0 + p1*x + p2*x^2 + p3*x^3
    D(x) = 1.0 + q1*x + q2*x^2
    
    params layout (length 10):
      params[0] = p0 (constant term in numerator)
      params[1] = p1 (linear term in numerator)
      params[2] = p2 (quadratic term in numerator)
      params[3] = p3 (cubic term in numerator)
      params[4] = q1 (linear term in denominator)
      params[5] = q2 (quadratic term in denominator)
      params[6] = p4 (quartic term in numerator)
      params[7] = p5 (quintic term in numerator)
      params[8] = q3 (cubic term in denominator)
      params[9] = q4 (quartic term in denominator)
    
    The denominator is bounded to stay positive for BFGS stability.
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Extract parameters
    p0, p1, p2, p3, p4, p5 = params[0], params[1], params[2], params[3], params[4], params[5]
    q1, q2, q3, q4 = params[6], params[7], params[8], params[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Numerator: degree 5 polynomial
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4 + p5 * x5
    
    # Denominator: 1 + degree 4 polynomial
    # Use tanh to bound the denominator perturbation to ensure positivity
    # D(x) = 1 + 0.3 * tanh(q1*x + q2*x2 + q3*x3 + q4*x4)
    # This ensures D(x) is in [0.7, 1.3], always positive
    denom_arg = q1 * x + q2 * x2 + q3 * x3 + q4 * x4
    denominator = 1.0 + 0.3 * np.tanh(denom_arg)
    
    # Compute result with safety against division by zero
    # Since denominator is in [0.7, 1.3], no division by zero possible
    result = numerator / denominator
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
