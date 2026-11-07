"""Symbolic regression with 1 input(s) - Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of Chebyshev polynomials of the first kind.
    
    We map x in [-2, 2] to t in [-1, 1] to use standard Chebyshev polynomials.
    This basis is orthogonal and well-conditioned, allowing BFGS to converge
    robustly to the least-squares solution.
    
    Basis functions:
      0: T_0(t) = 1
      1: T_1(t) = t
      2: T_2(t) = 2t^2 - 1
      ...
      9: T_9(t)
      
    where t = x / 2.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # Recurrence: T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, x.size), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
        
    # Linear combination
    # y = params[0]*T0 + params[1]*T1 + ...
    y = T.T @ params
    
    return y
# EVOLVE-BLOCK-END
