"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of a Fourier series basis scaled to [-2, 2].
    
    The basis consists of 10 sinusoidal terms with increasing frequencies,
    providing global periodicity and orthogonality properties.
    
    params: array of 10 coefficients.
    x: array of input points.
    """
    # Ensure x is float for calculation
    x = np.asarray(x, dtype=float)
    
    # Scale x to [-pi, pi] for standard Fourier frequencies
    # x in [-2, 2] -> t in [-pi, pi]
    t = np.pi * x / 2.0
    
    # Compute Fourier basis vectors: alternating sin and cos with increasing frequency
    # b1 = 1 (constant/DC term)
    # b2 = sin(t)
    # b3 = cos(t)
    # b4 = sin(2t)
    # b5 = cos(2t)
    # b6 = sin(3t)
    # b7 = cos(3t)
    # b8 = sin(4t)
    # b9 = cos(4t)
    # b10 = sin(5t)
    
    b1 = np.ones_like(x)
    b2 = np.sin(t)
    b3 = np.cos(t)
    b4 = np.sin(2.0 * t)
    b5 = np.cos(2.0 * t)
    b6 = np.sin(3.0 * t)
    b7 = np.cos(3.0 * t)
    b8 = np.sin(4.0 * t)
    b9 = np.cos(4.0 * t)
    b10 = np.sin(5.0 * t)
    
    # Linear combination
    result = (params[0] * b1 + 
              params[1] * b2 + 
              params[2] * b3 + 
              params[3] * b4 + 
              params[4] * b5 + 
              params[5] * b6 + 
              params[6] * b7 + 
              params[7] * b8 + 
              params[8] * b9 + 
              params[9] * b10)
              
    return result
# EVOLVE-BLOCK-END
