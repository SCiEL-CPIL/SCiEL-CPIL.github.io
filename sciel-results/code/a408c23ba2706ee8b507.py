"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1] range for better conditioning
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Full quadratic polynomial basis:
    # 1, u, v, w, u^2, v^2, w^2, uv, uw, vw
    # This utilizes all 10 parameters.
    
    term_0 = 1.0
    term_1 = u
    term_2 = v
    term_3 = w
    term_4 = u * u
    term_5 = v * v
    term_6 = w * w
    term_7 = u * v
    term_8 = u * w
    term_9 = v * w
    
    return (p[0] * term_0 + p[1] * term_1 + p[2] * term_2 + p[3] * term_3 +
            p[4] * term_4 + p[5] * term_5 + p[6] * term_6 + p[7] * term_7 +
            p[8] * term_8 + p[9] * term_9)
# EVOLVE-BLOCK-END
