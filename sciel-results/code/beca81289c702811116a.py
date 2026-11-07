"""Symbolic regression with a hybrid polynomial + trig + exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining polynomial, trigonometric, and exponential terms.
    
    y = a0 + a1*x + a2*x^2 + a3*x^3
      + b1*cos(x) + b2*sin(x) + b3*cos(2*x) + b4*sin(2*x)
      + c1*exp(-x^2) + c2*exp(-0.5*x^2)
    
    params layout (length 10):
      params[0..3]   = polynomial coefficients a0..a3
      params[4..7]   = trig coefficients b1..b4
      params[8..9]   = exponential coefficients c1, c2
    
    All coefficients are directly used (no tanh bounding) for better
    expressiveness. The basis functions are bounded on [-2,2].
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(params, dtype=float)

    # Polynomial basis
    a0, a1, a2, a3 = p[0], p[1], p[2], p[3]
    
    # Trig basis (using x directly, not pi*x, for different frequency)
    b1, b2, b3, b4 = p[4], p[5], p[6], p[7]
    
    # Exponential/Gaussian basis
    c1, c2 = p[8], p[9]
    
    # Compute basis values
    x2 = x * x
    x3 = x2 * x
    
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_2x = np.cos(2.0 * x)
    sin_2x = np.sin(2.0 * x)
    
    exp_neg_x2 = np.exp(-x2)
    exp_neg_half_x2 = np.exp(-0.5 * x2)
    
    result = (a0 + a1 * x + a2 * x2 + a3 * x3
              + b1 * cos_x + b2 * sin_x + b3 * cos_2x + b4 * sin_2x
              + c1 * exp_neg_x2 + c2 * exp_neg_half_x2)
    
    # Safety: guarantee finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
