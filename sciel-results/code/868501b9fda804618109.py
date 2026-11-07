import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a 9th-degree Chebyshev polynomial to the data.
    Basis: T_0(x/2), T_1(x/2), ..., T_9(x/2)
    This provides a well-conditioned linear basis for smooth functions on [-2, 2].
    """
    x = np.asarray(x, dtype=float)
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0 = 1
    # T_1 = t
    # T_{k+1} = 2*t*T_k - T_{k-1}
    
    T = np.empty((10, len(x)), dtype=float)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
            
    # The model is y = sum(p_i * T_i)
    # This is equivalent to T @ params
    return T @ params
