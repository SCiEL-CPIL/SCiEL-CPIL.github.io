"""
Symbolic regression with 2 inputs.
Uses a complete bivariate cubic polynomial basis (10 terms).
This basis is linear in parameters, allowing stable convergence with BFGS.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bivariate cubic polynomial basis.
    The model is:
    p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 + 
    p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    
    This basis has exactly 10 terms, matching the 10 parameters.
    It is linear in parameters, ensuring the optimization problem is convex
    and BFGS will converge to the global minimum for this basis.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Construct prediction
    # Order: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    pred = (p0 
            + p1 * x 
            + p2 * y 
            + p3 * x2 
            + p4 * x * y 
            + p5 * y2 
            + p6 * x3 
            + p7 * x2 * y 
            + p8 * x * y2 
            + p9 * y3)
            
    return pred
# EVOLVE-BLOCK-END
