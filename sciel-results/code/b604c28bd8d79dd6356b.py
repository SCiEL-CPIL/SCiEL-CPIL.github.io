"""
Symbolic regression using Chebyshev polynomial basis.
The hidden function is approximated by a linear combination of Chebyshev polynomials
of the first kind, scaled to the domain [-1, 1].
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Scale x from [-2, 2] to [-1, 1]
    # This aligns with the natural domain of Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Initialize the basis matrix
    # We need 10 basis functions corresponding to params[0..9]
    n_params = 10
    basis = np.zeros((len(x), n_params))
    
    if n_params > 0:
        basis[:, 0] = 1.0
    if n_params > 1:
        basis[:, 1] = t
        
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
