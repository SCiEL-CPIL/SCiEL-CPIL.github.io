"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1] for better conditioning
    # Inputs are in [0.5, 2.0], center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Full quadratic polynomial: 10 terms
    # p[0]: constant
    # p[1]: u, p[2]: v, p[3]: w
    # p[4]: u^2, p[5]: v^2, p[6]: w^2
    # p[7]: uv, p[8]: uw, p[9]: vw
    return (p[0] + p[1]*u + p[2]*v + p[3]*w +
            p[4]*u*u + p[5]*v*v + p[6]*w*w +
            p[7]*u*v + p[8]*u*w + p[9]*v*w)
# EVOLVE-BLOCK-END
