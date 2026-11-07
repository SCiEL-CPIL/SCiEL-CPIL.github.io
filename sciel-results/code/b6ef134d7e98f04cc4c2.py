"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with constant, linear, and sine/cosine terms up to k=4.
    params[0]  -> constant term
    params[1]  -> linear term (x)
    params[2]  -> sin(x)
    params[3]  -> cos(x)
    params[4]  -> sin(2x)
    params[5]  -> cos(2x)
    params[6]  -> sin(3x)
    params[7]  -> cos(3x)
    params[8]  -> sin(4x)
    params[9]  -> cos(4x)
    """
    # Ensure x is a numpy array for vectorized operations
    x = np.asarray(x, dtype=np.float64)
    
    # Initialize result with constant term
    result = np.full_like(x, params[0], dtype=np.float64)
    
    # Linear term
    result += params[1] * x
    
    # Frequency 1
    result += params[2] * np.sin(x)
    result += params[3] * np.cos(x)
    
    # Frequency 2
    result += params[4] * np.sin(2.0 * x)
    result += params[5] * np.cos(2.0 * x)
    
    # Frequency 3
    result += params[6] * np.sin(3.0 * x)
    result += params[7] * np.cos(3.0 * x)
    
    # Frequency 4
    result += params[8] * np.sin(4.0 * x)
    result += params[9] * np.cos(4.0 * x)
    
    return result
# EVOLVE-BLOCK-END
