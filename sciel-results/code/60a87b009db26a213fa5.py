"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a basis of polynomial monomials up to degree 3.
    This approach is distinct from trigonometric bases and is effective for
    capturing smooth, non-periodic structures in the bounded domain [-1, 1].
    """
    # Basis functions
    # Degree 0
    b0 = np.ones_like(x)
    
    # Degree 1
    b1 = x
    b2 = y
    
    # Degree 2
    b3 = x * x
    b4 = y * y
    b5 = x * y
    
    # Degree 3
    b6 = x * x * x
    b7 = y * y * y
    b8 = x * x * y
    b9 = x * y * y
    
    # Combine with parameters
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
