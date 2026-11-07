import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # We need 10 basis functions: T_0 ... T_9
    # Initialize arrays
    # To save memory and time, we can compute them in a loop or unroll.
    # Since N=10, unrolling or a small loop is fine.
    
    T_prev2 = np.ones_like(t)  # T_0
    T_prev1 = t.copy()          # T_1
    
    # Build basis matrix or compute weighted sum directly
    # y = p0*T0 + p1*T1 + p2*T2 + ... + p9*T9
    
    y = params[0] * T_prev2
    y += params[1] * T_prev1
    
    if len(params) > 2:
        T_curr = 2.0 * t * T_prev1 - T_prev2
        y += params[2] * T_curr
        
        T_prev2, T_prev1 = T_prev1, T_curr
        
        for i in range(3, min(10, len(params))):
            T_curr = 2.0 * t * T_prev1 - T_prev2
            y += params[i] * T_curr
            T_prev2, T_prev1 = T_prev1, T_curr
            
    return y
