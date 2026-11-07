"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a physically-motivated 10-term basis for nonlinear oscillator dynamics:
    - Restoring forces: x, x^3, x^5 (Duffing-type nonlinearity)
    - Damping forces: v, v^3, v^5 (van der Pol-type)
    - Cross-coupling: x*v, x^2*v, x*v^2
    - Constant offset
    """
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    v5 = v3 * v2
    
    p = params
    return (p[0] * x +
            p[1] * x3 +
            p[2] * x5 +
            p[3] * v +
            p[4] * v3 +
            p[5] * v5 +
            p[6] * x * v +
            p[7] * x2 * v +
            p[8] * x * v2 +
            p[9])
# EVOLVE-BLOCK-END
