"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed basis: Constant, Linear, Quadratic (squared), and Sine terms.
    This aims to capture both polynomial curvature and periodic/non-polynomial behavior.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    sin_z = np.sin(z)
    
    return (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * z + 
            p4 * x2 + 
            p5 * y2 + 
            p6 * z2 + 
            p7 * sin_x + 
            p8 * sin_y + 
            p9 * sin_z)
# EVOLVE-BLOCK-END
