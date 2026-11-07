"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a 2D polynomial basis up to degree 2,
    plus radial symmetry terms. This covers a broader class of functions
    than the linear model while keeping BFGS optimization well-conditioned.
    
    Basis:
    0: 1
    1: x
    2: y
    3: x^2
    4: xy
    5: y^2
    6: x^2 - y^2
    7: 2xy
    8: x^4 - 6x^2y^2 + y^4
    9: 4xy(x^2 - y^2)
    """
    # Ensure x and y are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Basic polynomial terms
    basis = np.stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        x * y,
        y**2,
        x**2 - y**2,
        2 * x * y,
        x**4 - 6 * x**2 * y**2 + y**4,
        4 * x * y * (x**2 - y**2),
    ], axis=-1)
    
    # Linear combination
    return np.einsum('ij,j->i', basis, params)
# EVOLVE-BLOCK-END
