"""Fourier series expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series expansion with multiple frequency components.
    
    Uses a combination of:
    - Constant and linear terms
    - sin/cos of x and y at multiple frequencies
    - sin/cos of x+y, x-y, and x*y at lower frequencies
    - Interaction terms
    
    params has 10 elements used as coefficients for the basis functions.
    """
    # Ensure params is 1D array
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Pad or truncate to exactly 10 params
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    else:
        p = p[:10]
    
    # Basis functions - mix of polynomial and Fourier terms
    # This gives flexibility for both smooth and oscillatory targets
    
    # Polynomial-like terms (for non-periodic structure)
    b0 = np.ones_like(x)                          # constant
    b1 = x                                        # linear x
    b2 = y                                        # linear y
    b3 = x * y                                    # interaction
    
    # Fourier terms for x
    b4 = np.sin(np.pi * x)                        # sin(pi*x)
    b5 = np.cos(np.pi * x)                        # cos(pi*x)
    
    # Fourier terms for y
    b6 = np.sin(np.pi * y)                        # sin(pi*y)
    b7 = np.cos(np.pi * y)                        # cos(pi*y)
    
    # Cross Fourier terms
    b8 = np.sin(np.pi * (x + y))                  # sin(pi*(x+y))
    b9 = np.sin(np.pi * (x - y))                  # sin(pi*(x-y))
    
    # Combine with learned coefficients
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
