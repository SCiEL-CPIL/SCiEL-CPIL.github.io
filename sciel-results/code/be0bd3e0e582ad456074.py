"""Symbolic regression: rational function with polynomial numerator and denominator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function: y = (p0 + p1*x + p2*x^2 + p3*x^3) / (1 + q0*x + q1*x^2 + q2*x^3)
    
    params[0:4] = numerator coefficients (p0, p1, p2, p3)
    params[4:7] = denominator coefficients (q0, q1, q2)
    params[7:10] = unused (set to 0 to avoid gradient issues)
    
    The denominator is guaranteed positive by using 1 + tanh-based regularization:
    denom = 1 + tanh(q0*x + q1*x^2 + q2*x^3)
    This ensures denom is in (0, 2), keeping the function well-behaved for BFGS.
    """
    x = np.asarray(x, dtype=float)
    
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    q0, q1, q2 = params[4], params[5], params[6]
    
    # Numerator: cubic polynomial
    x2 = x * x
    x3 = x2 * x
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3
    
    # Denominator: 1 + tanh(quadratic-in-x) ensures positivity and smoothness
    # tanh keeps values bounded, preventing division by zero
    denom_arg = q0 * x + q1 * x2 + q2 * x3
    denominator = 1.0 + np.tanh(denom_arg)
    
    # Guard against extremely small denominators (though tanh ensures >= 0)
    # tanh is in (-1, 1), so denominator is in (0, 2)
    # Add a small epsilon to be safe
    denominator = np.maximum(denominator, 1e-8)
    
    result = numerator / denominator
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
