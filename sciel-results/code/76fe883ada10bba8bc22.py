import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    n_basis = 10
    # Initialize basis matrix
    # P[:, 0] will be T_0
    # P[:, 1] will be T_1
    # ...
    # P[:, 9] will be T_9
    
    P = np.zeros((x.shape[0], n_basis))
    
    P[:, 0] = 1.0
    if n_basis > 1:
        P[:, 1] = t
        
    for k in range(2, n_basis):
        P[:, k] = 2.0 * t * P[:, k-1] - P[:, k-2]
        
    prediction = P @ params
    
    return prediction
