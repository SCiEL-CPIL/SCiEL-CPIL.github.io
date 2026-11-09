"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis.
    
    The domain x is in [-2, 2]. We map it to t in [-1, 1] using t = x/2.
    We use Chebyshev polynomials T_0(t) to T_9(t) as the basis functions.
    This basis is orthogonal on [-1, 1], leading to a well-conditioned
    optimization problem for the coefficients (params).
    
    params: array of 10 coefficients [c_0, ..., c_9]
    """
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_k(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    # Initialize basis array
    # Shape: (N, 10) where N is number of points
    n_points = len(x)
    basis = np.zeros((n_points, 10))
    
    if n_points == 0:
        return np.array([])

    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    if len(params) > 2:
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Compute prediction: sum(c_k * T_k(t))
    # Efficiently: basis @ params
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
