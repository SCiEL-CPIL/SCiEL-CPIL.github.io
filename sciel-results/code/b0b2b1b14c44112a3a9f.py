"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rich basis of functions.
    
    Basis functions:
    1. 1
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. sin(x)
    7. cos(x)
    8. sin(2x)
    9. cos(2x)
    10. exp(-x^2)
    
    This combination allows the model to fit polynomials up to degree 4,
    low-frequency sinusoidal components, and a central Gaussian bump.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # Ensure x is an array for consistent operations
    x = np.asarray(x)
    
    term0 = 1.0
    term1 = x
    term2 = x ** 2
    term3 = x ** 3
    term4 = x ** 4
    
    term5 = np.sin(x)
    term6 = np.cos(x)
    term7 = np.sin(2.0 * x)
    term8 = np.cos(2.0 * x)
    
    term9 = np.exp(-x ** 2)
    
    # Combine
    result = (p0 * term0 + 
              p1 * term1 + 
              p2 * term2 + 
              p3 * term3 + 
              p4 * term4 + 
              p5 * term5 + 
              p6 * term6 + 
              p7 * term7 + 
              p8 * term8 + 
              p9 * term9)
              
    return result
# EVOLVE-BLOCK-END
