import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    High-order Chebyshev polynomial basis.
    params[0:10] are coefficients for Chebyshev polynomials T_0 to T_9.
    This basis is orthogonal on [-1, 1], which helps BFGS converge.
    """
    if len(params) == 0:
        return np.zeros_like(x)
    
    # Map x from [-2, 2] to u in [-1, 1]
    u = x / 2.0
    
    # Initialize Chebyshev polynomials
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    n_params = len(params)
    # We have 10 parameters, so we can go up to T_9
    
    # Pre-compute Chebyshev basis vectors
    # We only need up to the number of parameters provided
    max_k = min(n_params, 10)
    
    if max_k == 0:
        return np.zeros_like(x)
        
    T_prev = np.ones_like(u)  # T_0
    if max_k == 1:
        return params[0] * T_prev
        
    T_curr = u  # T_1
    res = params[0] * T_prev + params[1] * T_curr
    
    for k in range(2, max_k):
        T_next = 2 * u * T_curr - T_prev
        res += params[k] * T_next
        T_prev = T_curr
        T_curr = T_next
        
    return res
