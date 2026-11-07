"""Symbolic regression using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Chebyshev polynomials of the first kind.
    
    y = sum_{k=0}^{9} p_k * T_k(x/2)
    
    where T_k is the k-th Chebyshev polynomial.
    The input x is in [-2, 2], so x/2 is in [-1, 1].
    
    This formulation is linear in the parameters, making the MSE landscape
    convex and allowing the optimizer (BFGS) to converge quickly and reliably
    to the global minimum. The Chebyshev basis is well-conditioned compared
    to monomials, improving numerical stability.
    
    params layout (length 10):
      params[0..9] = coefficients for T_0 to T_9
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2t * T_{n-1}(t) - T_{n-2}(t)
    
    # We only need to evaluate up to T_9
    # Storing them in a list or computing sequentially
    T = np.zeros((10, len(t)))
    
    T[0] = 1.0
    if len(t) > 0:
        T[1] = t
        for k in range(2, 10):
            T[k] = 2.0 * t * T[k-1] - T[k-2]
            
    # Calculate the linear combination
    # result = sum_{k=0}^{9} params[k] * T[k]
    # Using matrix multiplication for efficiency: T @ params
    # T is (10, N), params is (10,). Result is (N,)
    
    result = T @ params
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
