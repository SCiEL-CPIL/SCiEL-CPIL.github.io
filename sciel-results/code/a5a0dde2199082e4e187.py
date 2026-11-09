"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import numpy.polynomial.legendre as npleg
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate a hidden function using a combined basis of Legendre polynomials
    and trigonometric functions.
    
    The first 7 parameters correspond to Legendre polynomials P_0 through P_6.
    The remaining 3 parameters correspond to sin(x), cos(x), and sin(2x).
    
    Parameters:
    x : np.ndarray
        Input points.
    params : np.ndarray
        Array of 10 parameters to be fitted.
        
    Returns:
    np.ndarray
        Predicted values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Scale x to [-1, 1] for Legendre polynomials
    u = x / 3.0
    
    # Compute Legendre polynomial values P_0(u) to P_6(u)
    # legvander returns a matrix of shape (N, 7) where column k is P_k(u)
    V_leg = npleg.legvander(u, 6)
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    # Combine all basis functions
    # V_leg is (N, 7), we add 3 more columns
    V = np.empty((n, 10), dtype=np.float64)
    V[:, :7] = V_leg
    V[:, 7] = sin_x
    V[:, 8] = cos_x
    V[:, 9] = sin_2x
    
    # Linear combination of basis functions
    return V @ params
# EVOLVE-BLOCK-END
