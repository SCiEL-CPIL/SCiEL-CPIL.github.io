"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic pure-power polynomial basis in normalized coordinates.
    Normalizes inputs from [0.5, 2.0] to [-1, 1] for better conditioning.
    Terms: 1, u, v, w, u^2, v^2, w^2, u^3, v^3, w^3
    where u=(x-1.25)/0.75, v=(y-1.25)/0.75, w=(z-1.25)/0.75
    """
    p = params
    
    # Normalize inputs to [-1, 1] for better numerical conditioning
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    return (p[0] +
            p[1] * u +
            p[2] * v +
            p[3] * w +
            p[4] * u * u +
            p[5] * v * v +
            p[6] * w * w +
            p[7] * u * u * u +
            p[8] * v * v * v +
            p[9] * w * w * w)
# EVOLVE-BLOCK-END
