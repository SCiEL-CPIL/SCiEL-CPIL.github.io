"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis:
    f(x,y,z) = (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z)
    
    This uses 6 parameters for the product factors.
    We use the remaining 4 parameters to add a linear residual term:
    + d0 + d1*x + d2*y + d3*z
    
    Total: 10 parameters.
    """
    # params[0:3] -> a0, a1, a2 for x-factor
    # params[3:5] -> b0, b1 for y-factor
    # params[5:7] -> c0, c1 for z-factor
    # params[7:10] -> d0, d1, d2, d3 for linear residual
    
    a0, a1, a2 = params[0], params[1], params[2]
    b0, b1 = params[3], params[4]
    c0, c1 = params[5], params[6]
    d0, d1, d2, d3 = params[7], params[8], params[9]
    
    # Compute the product term
    fx = a0 + a1 * x + a2 * x**2
    fy = b0 + b1 * y
    fz = c0 + c1 * z
    product_term = fx * fy * fz
    
    # Compute the linear residual
    linear_term = d0 + d1 * x + d2 * y + d3 * z
    
    return product_term + linear_term
# EVOLVE-BLOCK-END
