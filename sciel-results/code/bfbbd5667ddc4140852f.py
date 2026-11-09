import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # We can compute these iteratively
    # Or use numpy? No direct built-in for Chebyshev values, but we can do it manually.
    
    T = np.zeros((len(x), 10))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2 * t * T[:, k-1] - T[:, k-2]
        
    return T @ params
