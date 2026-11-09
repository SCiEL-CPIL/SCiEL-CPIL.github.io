"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Trigonometric Basis Approximation
    # Uses a linear combination of sine and cosine functions of inputs and interactions.
    # This is a distinct approach from rational or polynomial bases.
    
    p = params
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-3. x terms
    b1 = np.sin(x)
    b2 = np.cos(x)
    
    # 4-5. y terms
    b3 = np.sin(y)
    b4 = np.cos(y)
    
    # 6-7. z terms
    b5 = np.sin(z)
    b6 = np.cos(z)
    
    # 8-9. x+y interaction
    b7 = np.sin(x + y)
    b8 = np.cos(x + y)
    
    # 10. x*z interaction
    b9 = np.sin(x * z)
    
    # Linear combination
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
              
    return result
# EVOLVE-BLOCK-END
