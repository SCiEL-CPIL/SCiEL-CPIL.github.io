"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich polynomial + trigonometric basis with 10 learnable parameters.
    Uses a combination of polynomial terms and trigonometric functions to
    span a wider space of possible hidden functions.
    """
    # Build feature matrix from 10 parameters
    # Features chosen to cover common function families:
    # polynomials (x, y, x*y, x^2, y^2), trig (sin, cos), and interactions
    
    f0 = np.ones_like(x, dtype=float)          # constant
    f1 = x                                      # linear x
    f2 = y                                      # linear y
    f3 = x * y                                  # interaction
    f4 = x ** 2                                 # quadratic x
    f5 = y ** 2                                 # quadratic y
    f6 = np.sin(x)                              # trig x
    f7 = np.sin(y)                              # trig y
    f8 = np.cos(x)                              # trig x
    f9 = np.cos(y)                              # trig y
    
    return (params[0] * f0 + params[1] * f1 + params[2] * f2 + 
            params[3] * f3 + params[4] * f4 + params[5] * f5 +
            params[6] * f6 + params[7] * f7 + params[8] * f8 + 
            params[9] * f9)
# EVOLVE-BLOCK-END
