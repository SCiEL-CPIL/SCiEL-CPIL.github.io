"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a mixed basis of polynomial, exponential, trigonometric, and hyperbolic functions.
    # This approach family is distinct from pure polynomial/Chebyshev bases and
    # allows for direct representation of common non-polynomial functional forms.
    
    x = np.asarray(x).flatten()
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = np.exp(x)
    b6 = np.exp(-x)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.tanh(x)
    
    # Stack into a matrix (N, 10)
    A = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination
    return A @ params
# EVOLVE-BLOCK-END
