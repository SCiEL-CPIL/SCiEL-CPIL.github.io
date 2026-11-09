import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # Using the recurrence relation for stability and speed
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    n_terms = 10
    V = np.zeros((len(x), n_terms), dtype=np.float64)
    
    V[:, 0] = 1.0
    if n_terms > 1:
        V[:, 1] = t
        for k in range(2, n_terms):
            V[:, k] = 2.0 * t * V[:, k-1] - V[:, k-2]
            
    return V @ p
