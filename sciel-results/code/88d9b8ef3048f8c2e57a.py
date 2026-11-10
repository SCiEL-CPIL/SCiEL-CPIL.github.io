"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a rich basis of polynomial and trigonometric terms.
    
    params is an array of length 10. We use all 10 parameters as coefficients for:
    1. Constant term
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. x^3
    8. y^3
    9. sin(x)
    10. cos(y)
    
    This provides a flexible basis that can approximate many smooth functions.
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions: Full degree-3 polynomial basis
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x**2
    term4 = y**2
    term5 = x * y
    term6 = x**3
    term7 = y**3
    term8 = x**2 * y
    term9 = x * y**2
    
    # Combine with parameters
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + 
            p8 * term8 + p9 * term9)
    
    return pred
# EVOLVE-BLOCK-END
