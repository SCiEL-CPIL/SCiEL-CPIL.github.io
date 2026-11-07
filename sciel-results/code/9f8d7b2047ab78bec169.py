"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes a general quadratic polynomial in normalized x, y, z.
    Normalization maps [0.5, 2] to [-1, 1] to improve conditioning for BFGS.
    
    Basis:
    0: 1
    1: u_x
    2: u_y
    3: u_z
    4: u_x^2
    5: u_y^2
    6: u_z^2
    7: u_x * u_y
    8: u_x * u_z
    9: u_y * u_z
    """
    p = params
    
    # Normalize inputs to [-1, 1]
    # Domain is [0.5, 2.0]. Center 1.25, Range 1.5.
    # u = (val - 1.25) / 0.75
    # This maps 0.5 -> -1, 2.0 -> 1.
    u_x = (x - 1.25) / 0.75
    u_y = (y - 1.25) / 0.75
    u_z = (z - 1.25) / 0.75
    
    return (p[0] + 
            p[1] * u_x + 
            p[2] * u_y + 
            p[3] * u_z + 
            p[4] * u_x * u_x + 
            p[5] * u_y * u_y + 
            p[6] * u_z * u_z + 
            p[7] * u_x * u_y + 
            p[8] * u_x * u_z + 
            p[9] * u_y * u_z)
# EVOLVE-BLOCK-END
