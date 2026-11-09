"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions chosen to cover a wide range of common symbolic regression targets
    # Replaced sin(2x), cos(2x) with x^4 and exp(x) to cover higher-order polynomials and exponential growth
    
    b1 = np.ones_like(x)
    b2 = x
    b3 = x ** 2
    b4 = x ** 3
    b5 = x ** 4
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.exp(-x ** 2)
    b9 = np.tanh(x)
    b10 = np.exp(x)
    
    # Linear combination of basis functions weighted by params
    y = (params[0] * b1 + 
         params[1] * b2 + 
         params[2] * b3 + 
         params[3] * b4 + 
         params[4] * b5 + 
         params[5] * b6 + 
         params[6] * b7 + 
         params[7] * b8 + 
         params[8] * b9 + 
         params[9] * b10)
         
    return y
# EVOLVE-BLOCK-END
