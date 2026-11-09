"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a rich basis of 10 features.
    x is scaled to u in [-1, 1] to improve numerical stability for BFGS.
    Basis:
    1. Constant
    2. u
    3. u^2
    4. u^3
    5. u^4
    6. exp(u)
    7. exp(-u)
    8. sin(u)
    9. cos(u)
    10. sin(2u)
    """
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    # u = (x - 3.05) / 2.95
    u = (x - 3.05) / 2.95
    
    # Construct basis terms
    b0 = np.ones_like(x)
    b1 = u
    b2 = u * u
    b3 = b2 * u
    b4 = b3 * u
    b5 = np.exp(u)
    b6 = np.exp(-u)
    b7 = np.sin(u)
    b8 = np.cos(u)
    b9 = np.sin(2 * u)
    
    # Combine with params
    return (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
# EVOLVE-BLOCK-END
