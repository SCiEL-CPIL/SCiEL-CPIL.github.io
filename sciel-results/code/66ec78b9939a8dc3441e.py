"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-pi, pi] for Fourier basis
    theta = x * (np.pi / 3.0)
    
    # Basis functions:
    # p[0]: constant
    # p[1]: cos(theta)
    # p[2]: sin(theta)
    # p[3]: cos(2*theta)
    # p[4]: sin(2*theta)
    # p[5]: cos(3*theta)
    # p[6]: sin(3*theta)
    # p[7]: cos(4*theta)
    # p[8]: sin(4*theta)
    # p[9]: cos(5*theta)
    
    basis = np.ones((len(x), 10))
    
    # Frequencies 1 to 4 have both sin and cos
    for k in range(1, 5):
        basis[:, 1 + 2*(k-1)] = np.cos(k * theta)
        basis[:, 2 + 2*(k-1)] = np.sin(k * theta)
        
    # Frequency 5 has only cos
    basis[:, 9] = np.cos(5 * theta)
    
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
