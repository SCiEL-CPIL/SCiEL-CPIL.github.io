"""Symbolic regression: recover an unknown function with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Return predictions for input x using 10 parameters.
    
    Uses a rich basis: polynomial (deg 0-4), sin, cos, exp, and 1/x terms.
    Basis: [1, x, x^2, x^3, x^4, sin(x), cos(x), exp(x), exp(-x), 1/x]
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Clamp x to avoid extreme values (though x is in [0.1, 6])
    x_safe = np.clip(x, 1e-6, 10.0)
    
    # Compute basis functions
    x2 = x_safe * x_safe
    x3 = x2 * x_safe
    x4 = x2 * x2
    
    sin_x = np.sin(x_safe)
    cos_x = np.cos(x_safe)
    
    # exp(x) can be large for x=6 (exp(6)~403), but manageable
    exp_x = np.exp(x_safe)
    exp_neg_x = np.exp(-x_safe)
    
    inv_x = 1.0 / x_safe
    
    # Combine: y = p0*1 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 
    #        + p5*sin(x) + p6*cos(x) + p7*exp(x) + p8*exp(-x) + p9*(1/x)
    y = (p[0] * 1.0 +
         p[1] * x_safe +
         p[2] * x2 +
         p[3] * x3 +
         p[4] * x4 +
         p[5] * sin_x +
         p[6] * cos_x +
         p[7] * exp_x +
         p[8] * exp_neg_x +
         p[9] * inv_x)
    
    return y
# EVOLVE-BLOCK-END
