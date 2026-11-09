"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a Fourier series to the data.
    Input x is in [-2, 2].
    We use a basis of sine and cosine functions with frequencies adapted to the domain length 4.
    Basis:
    p[0]: 1
    p[1]: cos(pi*x/2), p[2]: sin(pi*x/2)
    p[3]: cos(pi*x), p[4]: sin(pi*x)
    p[5]: cos(3*pi*x/2), p[6]: sin(3*pi*x/2)
    p[7]: cos(2*pi*x), p[8]: sin(2*pi*x)
    p[9]: cos(5*pi*x/2)
    """
    # Calculate basis terms
    # Note: Using np.cos and np.sin directly
    
    result = params[0] * np.ones_like(x)
    
    # Frequency 1: pi/2
    c1 = np.cos(np.pi * x / 2.0)
    s1 = np.sin(np.pi * x / 2.0)
    result += params[1] * c1 + params[2] * s1
    
    # Frequency 2: pi
    c2 = np.cos(np.pi * x)
    s2 = np.sin(np.pi * x)
    result += params[3] * c2 + params[4] * s2
    
    # Frequency 3: 3*pi/2
    c3 = np.cos(3.0 * np.pi * x / 2.0)
    s3 = np.sin(3.0 * np.pi * x / 2.0)
    result += params[5] * c3 + params[6] * s3
    
    # Frequency 4: 2*pi
    c4 = np.cos(2.0 * np.pi * x)
    s4 = np.sin(2.0 * np.pi * x)
    result += params[7] * c4 + params[8] * s4
    
    # Frequency 5: 5*pi/2 (using only cosine to keep 10 params)
    c5 = np.cos(5.0 * np.pi * x / 2.0)
    result += params[9] * c5
    
    return result
# EVOLVE-BLOCK-END
