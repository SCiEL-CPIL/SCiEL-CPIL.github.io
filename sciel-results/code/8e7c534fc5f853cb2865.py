"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Cubic polynomial basis in x and y
    # Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    # This uses exactly 10 parameters, matching the required length.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    return (p0 + p1 * x + p2 * y + 
            p3 * x**2 + p4 * x * y + p5 * y**2 + 
            p6 * x**3 + p7 * x**2 * y + p8 * x * y**2 + p9 * y**3)
# EVOLVE-BLOCK-END
