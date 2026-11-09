"""
Strategy: Hybrid Rational Function Basis.

This approach uses a rational function (ratio of two low-degree polynomials)
with learnable coefficients. Rational functions are more flexible than pure
polynomials for capturing features like hyperbolic behavior, sharp transitions,
and localized structures, while maintaining numerical stability when the
denominator is regularized.

The model is:
y(x) = offset + (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1.0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)

Parameters (length 10):
- params[0:5]: a0, a1, a2, a3, a4 (numerator coefficients)
- params[5:9]: b1, b2, b3, b4 (denominator coefficients, constant term fixed to 1)
- params[9]: offset (global bias)

This is materially different from the previous hinge basis approach. Rational
functions can exactly represent hyperbolic functions, approximate step functions
with appropriate pole placement, and capture complex smooth behaviors that
polynomials or piecewise linear functions approximate poorly.
"""

import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Tunable Rational Function Basis.
    
    y(x) = offset + P(x) / Q(x)
    where P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    and Q(x) = 1.0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 parameters [a0, a1, a2, a3, a4, b1, b2, b3, b4, offset]
    """
    # Unpack parameters
    a0, a1, a2, a3, a4, b1, b2, b3, b4, offset = params
    
    # Compute Numerator: P(x) using Horner's method
    num = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
    
    # Compute Denominator: Q(x) using Horner's method
    # Constant term fixed to 1.0 for stability
    den = 1.0 + x * (b1 + x * (b2 + x * (b3 + x * b4)))
    
    # Safety: prevent division by zero or near-zero
    # Add small epsilon to avoid numerical instability
    den_safe = den + 1e-9
    
    result = offset + num / den_safe
    
    return result
