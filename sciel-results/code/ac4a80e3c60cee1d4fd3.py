"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    Basis: 1, x, x^2, x^3, v, v^2, xv, x^2v, cos(t), sin(t)
    """
    p = params
    
    # Features
    f0 = 1.0
    f1 = x
    f2 = x * x
    f3 = x * x * x
    f4 = v
    f5 = v * v
    f6 = x * v
    f7 = x * x * v
    f8 = np.cos(t)
    f9 = np.sin(t)
    
    # Linear combination
    y = (p[0] * f0 + 
         p[1] * f1 + 
         p[2] * f2 + 
         p[3] * f3 + 
         p[4] * f4 + 
         p[5] * f5 + 
         p[6] * f6 + 
         p[7] * f7 + 
         p[8] * f8 + 
         p[9] * f9)
         
    return y

# EVOLVE-BLOCK-END
