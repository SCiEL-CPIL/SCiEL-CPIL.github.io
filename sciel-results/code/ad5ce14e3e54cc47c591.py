"""Symbolic regression with 1 input using a well-conditioned hybrid basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 basis functions.
    The basis is a mix of scaled polynomials, trigonometric functions, and an exponential,
    designed to cover a wide range of potential target functions on [0.1, 6].
    
    The input x is scaled to u in [0, 1] to improve numerical conditioning for the optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Scale x from [0.1, 6.0] to u in [0.0, 1.0]
    x_min = 0.1
    x_max = 6.0
    u = (x - x_min) / (x_max - x_min)
    
    # Define 10 basis functions
    # Polynomials (scaled)
    phi_0 = np.ones_like(u)
    phi_1 = u
    phi_2 = u**2
    phi_3 = u**3
    phi_4 = u**4
    
    # Trigonometric (scaled to [0, 2pi] and [0, 4pi])
    # Using 2*pi*u ensures integer periods over the domain, which is often helpful
    phi_5 = np.sin(2.0 * np.pi * u)
    phi_6 = np.cos(2.0 * np.pi * u)
    phi_7 = np.sin(4.0 * np.pi * u)
    phi_8 = np.cos(4.0 * np.pi * u)
    
    # Exponential decay
    phi_9 = np.exp(-u)
    
    # Stack into a matrix of shape (N, 10)
    basis = np.column_stack((
        phi_0, phi_1, phi_2, phi_3, phi_4,
        phi_5, phi_6, phi_7, phi_8, phi_9
    ))
    
    # The prediction is the dot product of the basis matrix and the parameters
    # basis: (N, 10), params: (10,) -> result: (N,)
    return basis @ params
# EVOLVE-BLOCK-END
