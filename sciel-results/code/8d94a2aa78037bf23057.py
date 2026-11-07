"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis mixing polynomial and nonlinear terms.
    Combines the cubic polynomial structure of the current program with
    nonlinear terms (sin, tanh) from inspirations to capture a wider range
    of possible hidden functions.
    
    Basis: 1, x, y, x^2, y^2, x*y, sin(x), sin(y), tanh(x), tanh(y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Nonlinear terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    tanh_x = np.tanh(x)
    tanh_y = np.tanh(y)
    
    return (p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy +
            p6 * sin_x + p7 * sin_y + p8 * tanh_x + p9 * tanh_y)
# EVOLVE-BLOCK-END
