"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import numpy.polynomial.legendre as npleg


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate a hidden function using an orthogonal polynomial basis (Legendre).
    This basis is well-conditioned and can fit any smooth function up to degree 9.
    
    Parameters:
    x : np.ndarray
        Input points.
    params : np.ndarray
        Array of 10 parameters to be fitted.
        
    Returns:
    np.ndarray
        Predicted values.
    """
    # Scale x to [-1, 1] for Legendre polynomials
    u = x / 3.0
    
    # Compute Legendre polynomial values P_0(u) to P_9(u)
    # legvander returns a matrix of shape (N, 10) where column k is P_k(u)
    V = npleg.legvander(u, 9)
    
    # Linear combination of basis functions
    return V @ params
# EVOLVE-BLOCK-END
