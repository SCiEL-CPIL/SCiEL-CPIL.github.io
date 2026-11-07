"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric basis: product of 1D Fourier-like terms
    # This allows modeling periodic/oscillatory functions
    # Features: 1, cos(x), sin(x), cos(y), sin(y), cos(x)*sin(y), 
    #           cos(2x), sin(2y), cos(x)*cos(y), sin(x)*sin(y)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    # Second harmonic terms
    cos_2x = np.cos(2 * x)
    sin_2y = np.sin(2 * y)
    
    # Interaction terms
    cos_x_sin_y = cos_x * sin_y
    cos_x_cos_y = cos_x * cos_y
    sin_x_sin_y = sin_x * sin_y
    
    return (p0 + 
            p1 * cos_x + 
            p2 * sin_x + 
            p3 * cos_y + 
            p4 * sin_y + 
            p5 * cos_x_sin_y + 
            p6 * cos_2x + 
            p7 * sin_2y + 
            p8 * cos_x_cos_y + 
            p9 * sin_x_sin_y)
# EVOLVE-BLOCK-END
