"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] to improve numerical stability for polynomial fitting
    u = x / 3.0
    # Construct a polynomial of degree 9
    # y = p0 + p1*u + ... + p9*u^9
    return (params[0] + 
            params[1] * u + 
            params[2] * u**2 + 
            params[3] * u**3 + 
            params[4] * u**4 + 
            params[5] * u**5 + 
            params[6] * u**6 + 
            params[7] * u**7 + 
            params[8] * u**8 + 
            params[9] * u**9)
# EVOLVE-BLOCK-END
