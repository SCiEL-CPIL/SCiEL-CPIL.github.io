"""Rational function approximation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = P(x) / (Q(x) + 1)
    
    P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    Q(x) = q0*x + q1*x^2 + q2*x^3 + q3*x^4 + q5*x^5
    
    params layout (length 10):
      params[0..4] = p0, p1, p2, p3, p4
      params[5..9] = q0, q1, q2, q3, q4 (q4 used as q5 for x^5 term)
    
    The +1 in the denominator ensures it is positive near x=0 when params start at 1,
    helping BFGS convergence. We use a soft regularization to keep Q away from -1.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    q0, q1, q2, q3, q4 = params[5], params[6], params[7], params[8], params[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Numerator: P(x)
    P = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4
    
    # Denominator: Q(x) + 1
    Q = q0 * x + q1 * x2 + q2 * x3 + q3 * x4 + q4 * x5
    D = Q + 1.0
    
    # Ensure denominator doesn't get too close to zero for numerical stability
    # Use a soft floor: if |D| < eps, replace with sign(D)*eps
    eps = 1e-8
    D_safe = np.where(np.abs(D) < eps, np.sign(D) * eps, D)
    # Handle case where D is exactly 0
    D_safe = np.where(D_safe == 0, eps, D_safe)
    
    result = P / D_safe
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
