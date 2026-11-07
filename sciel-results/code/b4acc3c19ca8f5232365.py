"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Chebyshev polynomials to approximate the target.
    Maps x in [-2, 2] to t in [-1, 1] and uses T_0(t) to T_9(t).
    This basis is well-conditioned for BFGS optimization.
    """
    p = params
    
    # Ensure finite params to avoid NaN/Inf during optimization steps
    if not np.all(np.isfinite(p)):
        p = np.ones(10)

    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    # Use a stable iterative computation
    # T is shape (10, N) where N is len(x)
    N = x.shape[0]
    T = np.zeros((10, N))
    
    T[0, :] = 1.0
    if N > 0:
        T[1, :] = t
        
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
            
    # Compute prediction: sum(p_i * T_i(t))
    # T is (10, N), p is (10,)
    pred = np.dot(p, T)
    
    # Safety: replace any non-finite values with 0
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
