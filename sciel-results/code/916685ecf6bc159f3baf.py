"""
Chebyshev basis strategy for symbolic regression.
Uses Chebyshev polynomials of the first kind mapped to [-2, 2].
This basis is well-conditioned for BFGS and provides optimal approximation
properties for smooth functions on the given domain.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of Chebyshev polynomials.
    
    Args:
        x: Input array of shape (N,)
        params: Array of 10 coefficients
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev polynomials
    # t = (x - 0) / 2 = x / 2
    t = x / 2.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)), dtype=float)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Compute linear combination: sum(params[i] * T[i](x))
    # params is shape (10,), T is shape (10, N)
    # Result is shape (N,)
    prediction = params @ T
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
