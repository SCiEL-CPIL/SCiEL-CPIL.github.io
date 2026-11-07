import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    # We will accumulate the result
    result = np.zeros_like(x)
    
    if len(params) == 0:
        return result
        
    # T_0
    T_k_minus_2 = np.ones_like(x)
    result += params[0] * T_k_minus_2
    
    if len(params) > 1:
        # T_1
        T_k_minus_1 = t
        result += params[1] * T_k_minus_1
        
        for k in range(2, len(params)):
            T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
            result += params[k] * T_k
            T_k_minus_2 = T_k_minus_1
            T_k_minus_1 = T_k
            
    return result
