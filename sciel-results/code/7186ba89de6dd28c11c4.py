"""Symbolic regression using Chebyshev polynomial basis functions."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate y(x) using a sum of the first 10 Chebyshev polynomials of the first kind.
    
    The domain is x in [-2, 2]. We map this to t in [-1, 1] by t = x / 2.
    The model is:
    y(x) = sum_{k=0}^{9} params[k] * T_k(x / 2)
    
    Chebyshev polynomials provide a stable, well-conditioned basis for smooth
    functions, allowing BFGS to converge rapidly to the optimal coefficients.
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.empty((10, x.size), dtype=float)
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # The prediction is the dot product of the design matrix T (transposed) and params
    # T is (10, N), params is (10,). Result is (N,).
    result = T.T @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
