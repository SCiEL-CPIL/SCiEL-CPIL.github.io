"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis of degree 9.
    
    The model is:
    f(x) = sum_{k=0}^{9} c_k * T_k(x/2)
    
    where T_k is the k-th Chebyshev polynomial of the first kind.
    This basis is well-conditioned on the domain [-2, 2] (mapped to [-1, 1])
    and allows BFGS to efficiently fit smooth functions.
    """
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Precompute the basis functions T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Initialize basis array with shape (N, 10)
    N = len(x)
    basis = np.zeros((N, 10))
    
    # T_0
    basis[:, 0] = 1.0
    
    # T_1
    if N > 0:
        basis[:, 1] = t
        
        # Recurrence for T_2 to T_9
        t_prev2 = basis[:, 0]  # T_0
        t_prev1 = basis[:, 1]  # T_1
        
        for k in range(2, 10):
            t_curr = 2.0 * t * t_prev1 - t_prev2
            basis[:, k] = t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr
    
    # The model is linear in params: y = basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
