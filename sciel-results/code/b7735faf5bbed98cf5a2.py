import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 through T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    T = np.zeros((n_points, 10))
    T[:, 0] = 1.0
    T[:, 1] = u
    
    for k in range(2, 10):
        T[:, k] = 2.0 * u * T[:, k-1] - T[:, k-2]
        
    # The model is a linear combination of these basis functions
    # y = sum(params[i] * T_i(x))
    return T @ params
