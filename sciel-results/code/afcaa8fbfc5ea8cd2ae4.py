"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute basis functions with improved scaling for BFGS conditioning
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    
    # Scaled polynomials (divide by factorial-like scaling to keep O(1))
    b0 = 1.0
    b1 = x / 2.0
    b2 = x2 / 6.0
    b3 = x3 / 18.0
    b4 = x4 / 54.0
    
    # Additional scaled polynomial terms
    b5 = x5 / 162.0
    b6 = x6 / 486.0
    
    # Trigonometric terms at multiple frequencies
    b7 = np.sin(x)
    b8 = np.cos(x)
    
    # Higher frequency trig terms
    b9 = np.sin(2.0 * x)
    
    # Gaussian bump
    b10 = np.exp(-x2 / 2.0)
    
    # Combine all 10 basis functions
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
