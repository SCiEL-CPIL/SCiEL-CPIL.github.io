"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Hybrid basis combining:
      - Polynomial terms (up to degree 5) for global smooth trends
      - tanh non-linearities at different scales for saturation/transition effects
      - A constant term
      
    This gives 10 parameters total.
    
    Basis functions:
      0: 1 (constant)
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: x^5
      6: tanh(x)
      7: tanh(2*x)
      8: tanh(0.5*x)
      9: tanh(4*x)
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = x ** 5
    
    # tanh non-linearities at different scales
    # Use np.clip to avoid overflow issues with tanh for very large arguments
    x_clipped = np.clip(x, -50, 50)
    b6 = np.tanh(x_clipped)
    b7 = np.tanh(2.0 * x_clipped)
    b8 = np.tanh(0.5 * x_clipped)
    b9 = np.tanh(4.0 * x_clipped)
    
    # Linear combination of basis functions with params
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
         params[3] * b3 + params[4] * b4 + params[5] * b5 +
         params[6] * b6 + params[7] * b7 + params[8] * b8 + 
         params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
