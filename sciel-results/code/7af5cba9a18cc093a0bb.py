"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Scale x from [-3,3] to [-1,1] for Legendre polynomials
    t = x / 3.0
    
    # Legendre polynomials P_0 through P_5 evaluated at t
    # P_0 = 1
    # P_1 = t
    # P_2 = (3t^2 - 1)/2
    # P_3 = (5t^3 - 3t)/2
    # P_4 = (35t^4 - 30t^2 + 3)/8
    # P_5 = (63t^5 - 70t^3 + 15t)/8
    
    P0 = np.ones_like(t)
    P1 = t
    P2 = (3.0 * t**2 - 1.0) / 2.0
    P3 = (5.0 * t**3 - 3.0 * t) / 2.0
    P4 = (35.0 * t**4 - 30.0 * t**2 + 3.0) / 8.0
    P5 = (63.0 * t**5 - 70.0 * t**3 + 15.0 * t) / 8.0
    
    # Trig terms
    s = np.sin(x)
    c = np.cos(x)
    xs = x * s
    xc = x * c
    
    # Build prediction: sum of param_i * basis_i
    pred = (params[0] * P0 +
            params[1] * P1 +
            params[2] * P2 +
            params[3] * P3 +
            params[4] * P4 +
            params[5] * P5 +
            params[6] * s +
            params[7] * c +
            params[8] * xs +
            params[9] * xc)
    
    return pred
# EVOLVE-BLOCK-END
