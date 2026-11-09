import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for better numerical stability
    t = x / 3.0
    
    # Precompute Chebyshev basis up to degree 9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    # To handle empty arrays or single points safely
    N = len(t)
    if N == 0:
        return np.array([])
        
    basis = np.zeros((N, 10))
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    return basis @ params
