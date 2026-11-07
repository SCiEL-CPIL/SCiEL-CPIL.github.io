"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis: polynomial + rational terms to span a wider function space.
    Uses all 10 params with stable features on [-1,1]^2.
    """
    # Polynomial features
    f0 = np.ones_like(x, dtype=float)       # constant
    f1 = x                                    # linear x
    f2 = y                                    # linear y
    f3 = x * y                                # interaction
    f4 = x ** 2                               # quadratic x
    f5 = y ** 2                               # quadratic y
    f6 = x ** 3                               # cubic x
    f7 = y ** 3                               # cubic y
    
    # Rational features (stable on [-1,1]: denominators in [1, 2])
    f8 = x / (1.0 + x ** 2)                   # odd rational, bounded
    f9 = y / (1.0 + y ** 2)                   # odd rational, bounded
    
    return (params[0] * f0 + params[1] * f1 + params[2] * f2 +
            params[3] * f3 + params[4] * f4 + params[5] * f5 +
            params[6] * f6 + params[7] * f7 +
            params[8] * f8 + params[9] * f9)
# EVOLVE-BLOCK-END
