"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] to improve numerical stability of the polynomial basis
    t = x / 3.0
    
    # Create a polynomial basis of degree 9 using scaled x
    # Basis: 1, t, t^2, ..., t^9
    # We can use np.vander or compute powers directly.
    # np.vander(t, N=10, increasing=True) gives columns [1, t, t^2, ..., t^9]
    basis = np.vander(t, N=10, increasing=True)
    
    # The model is a linear combination of the basis functions with coefficients params
    # y = params[0]*1 + params[1]*t + ... + params[9]*t^9
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
