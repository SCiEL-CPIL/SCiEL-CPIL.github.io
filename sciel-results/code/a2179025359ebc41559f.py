"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model: a9 + a8*x + a7*x^2 + a6*x^3 + a5*x^4 + a4*x^5
              + a3*x^6 + a2*x^7 + a1*x^8 + a0*x^9
    This is a 9th-degree polynomial with 10 free parameters, which can
    approximate a very wide range of smooth functions on [-2, 2].
    """
    x = np.asarray(x, dtype=float)
    p = params
    
    # Use Horner's method for numerical stability and efficiency
    # p[0] is coefficient of x^9, p[9] is constant term
    result = p[0]
    for i in range(1, 10):
        result = result * x + p[i]
    
    return result
# EVOLVE-BLOCK-END
