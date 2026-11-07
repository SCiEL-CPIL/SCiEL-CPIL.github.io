"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [0, 1] to standardize the frequency scale
    # x range is [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    u = (x - x_min) / (x_max - x_min)
    
    # Construct Fourier basis
    # params[0] is the constant term
    # params[1:9] are sin/cos for harmonics 1-4
    # params[9] is sin for harmonic 5
    # This uses 10 parameters total.
    
    # Precompute sine and cosine values for efficiency
    # We need harmonics 1, 2, 3, 4, 5
    # h1: 2*pi*u
    # h2: 4*pi*u
    # h3: 6*pi*u
    # h4: 8*pi*u
    # h5: 10*pi*u
    
    two_pi = 2 * np.pi
    
    h1 = two_pi * u
    h2 = 2 * h1
    h3 = 3 * h1
    h4 = 4 * h1
    h5 = 5 * h1
    
    # Basis vectors
    b0 = np.ones_like(u)
    b1 = np.sin(h1)
    b2 = np.cos(h1)
    b3 = np.sin(h2)
    b4 = np.cos(h2)
    b5 = np.sin(h3)
    b6 = np.cos(h3)
    b7 = np.sin(h4)
    b8 = np.cos(h4)
    b9 = np.sin(h5)
    
    # Combine with parameters
    # y = p0*b0 + p1*b1 + ... + p9*b9
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
