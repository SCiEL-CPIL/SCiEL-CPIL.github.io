"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a hybrid basis of low-degree polynomials and Fourier terms.
    
    The basis functions are:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: sin(3x)
    9: cos(3x)
    
    params[0..9] are the coefficients for these basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has at least 10 elements, pad with zeros if necessary
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)), mode='constant')
    
    # Compute basis functions
    # Polynomials
    p0 = np.ones_like(x)
    p1 = x
    p2 = x * x
    p3 = x * x * x
    
    # Fourier terms
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2.0 * x)
    c2 = np.cos(2.0 * x)
    s3 = np.sin(3.0 * x)
    c3 = np.cos(3.0 * x)
    
    # Combine linearly
    result = (params[0] * p0 + 
              params[1] * p1 + 
              params[2] * p2 + 
              params[3] * p3 + 
              params[4] * s1 + 
              params[5] * c1 + 
              params[6] * s2 + 
              params[7] * c2 + 
              params[8] * s3 + 
              params[9] * c3)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
