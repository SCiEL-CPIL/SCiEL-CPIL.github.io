import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis of degree 9.
    This basis is orthogonal on [-1, 1] and provides a stable representation
    for smooth functions, avoiding the optimization issues caused by mixing
    different function types (trig/exp/poly) which can lead to ill-conditioning.
    """
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Generate Chebyshev polynomials T_0 to T_9
    # Using the recurrence relation: T_{n+1}(t) = 2t * T_n(t) - T_{n-1}(t)
    T = np.zeros((10, x.shape[0]))
    
    T[0, :] = 1.0
    if x.shape[0] > 0:
        T[1, :] = t
        
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
            
    # Compute the linear combination
    # params has shape (10,), T has shape (10, N)
    # Result y has shape (N,)
    y = np.dot(params, T)
    
    return y
