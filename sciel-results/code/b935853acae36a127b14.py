"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a 9th-degree polynomial with coefficients given by params.
    This is a global approximation strategy, distinct from the local RBF approach.
    The basis functions are powers of x: 1, x, x^2, ..., x^9.
    BFGS will optimize the 10 coefficients to minimize training MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is 1D
    if x.ndim != 1:
        x = x.flatten()
        
    # Compute polynomial basis: [1, x, x^2, ..., x^9]
    # Using np.vander for efficient computation
    # np.vander(x, N=10, increasing=True) gives columns 1, x, x^2, ...
    basis = np.vander(x, N=10, increasing=True)
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
