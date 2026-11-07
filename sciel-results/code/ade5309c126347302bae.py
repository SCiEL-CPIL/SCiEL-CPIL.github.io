import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)))
    T[0] = 1.0
    if len(x) > 0:
        T[1] = t
    for i in range(2, 10):
        T[i] = 2.0 * t * T[i-1] - T[i-2]
        
    # Linear combination
    # params is shape (10,)
    # T is shape (10, N)
    # result is shape (N,)
    return np.dot(params, T)
