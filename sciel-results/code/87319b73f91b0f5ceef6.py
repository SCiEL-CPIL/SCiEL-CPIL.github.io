"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a combination of polynomial and 
    trigonometric basis functions.
    
    The model is:
    y = p0 + p1*x + p2*x^2 + p3*x^3 
      + p4*sin(x) + p5*cos(x) 
      + p6*sin(2x) + p7*cos(2x) 
      + p8*sin(3x) + p9*cos(3x)
      
    This provides flexibility for both global trends (polynomials) and 
    periodic/oscillatory components (trigonometrics) while remaining 
    differentiable and well-behaved for BFGS optimization.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis (up to cubic)
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Trigonometric basis (frequencies 1, 2, 3)
    trig = (p4 * np.sin(x) + p5 * np.cos(x) + 
            p6 * np.sin(2 * x) + p7 * np.cos(2 * x) + 
            p8 * np.sin(3 * x) + p9 * np.cos(3 * x))
    
    return poly + trig
# EVOLVE-BLOCK-END
