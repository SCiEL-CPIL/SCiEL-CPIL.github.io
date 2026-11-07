"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Legendre polynomial basis on [-2, 2] mapped to [-1, 1].
    
    Legendre polynomials are orthogonal with respect to the uniform weight function,
    which matches the uniform distribution of the input data on [-2, 2].
    This provides a well-conditioned basis for the linear least-squares problem,
    helping BFGS converge to the global optimum more reliably than Chebyshev
    or monomial bases which may be ill-conditioned or mismatched to the weight.
    
    We use 10 Legendre polynomials P_0 through P_9.
    """
    # Map from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Legendre polynomials P_k(t) for k=0..9
    # scipy.special.eval_legendre(n, x) computes P_n(x)
    # We stack them into a matrix of shape (n_points, 10)
    
    # Pre-allocate or list comprehension
    basis_list = []
    for k in range(10):
        basis_list.append(eval_legendre(k, t))
        
    basis = np.stack(basis_list, axis=-1)
    
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
