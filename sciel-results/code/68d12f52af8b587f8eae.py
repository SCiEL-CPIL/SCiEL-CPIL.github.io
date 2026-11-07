"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define a rich basis of functions to allow approximation of various target shapes.
    # The model is linear in params, ensuring the loss surface is convex (quadratic),
    # allowing BFGS to converge to the global minimum quickly.
    
    # 10 basis functions
    phi0 = np.ones_like(x)
    phi1 = x
    phi2 = x**2
    phi3 = x**3
    phi4 = x**4
    phi5 = x**5
    phi6 = np.sin(x)
    phi7 = np.cos(x)
    phi8 = np.sin(2*x)
    phi9 = np.cos(2*x)
    
    # Combine
    y = (params[0] * phi0 + 
         params[1] * phi1 + 
         params[2] * phi2 + 
         params[3] * phi3 + 
         params[4] * phi4 + 
         params[5] * phi5 + 
         params[6] * phi6 + 
         params[7] * phi7 + 
         params[8] * phi8 + 
         params[9] * phi9)
         
    return y
# EVOLVE-BLOCK-END
