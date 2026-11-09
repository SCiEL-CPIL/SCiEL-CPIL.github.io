import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use Chebyshev polynomials of the first kind as the basis.
    Map x from [0.1, 6] to u in [-1, 1].
    Compute T_0 to T_9.
    Return linear combination with params.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x to [-1, 1]
    # x_min = 0.1, x_max = 6.0
    # u = (2x - (x_max + x_min)) / (x_max - x_min)
    u = (2.0 * x - 6.1) / 5.9
    
    # Initialize output
    y = np.zeros_like(x)
    
    # T_0 = 1
    T_prev = np.ones_like(x)
    y += params[0] * T_prev
    
    if len(params) > 1:
        # T_1 = u
        T_curr = u
        y += params[1] * T_curr
        
        for k in range(2, 10):
            # T_k = 2u T_{k-1} - T_{k-2}
            T_next = 2.0 * u * T_curr - T_prev
            y += params[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return y
