import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis.
    
    Chebyshev polynomials are orthogonal on [-1, 1] and provide
    rapid convergence for smooth functions on bounded intervals.
    We map the input domain [-2, 2] to [-1, 1] and use degrees 0-9.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T_prev = np.ones_like(t)  # T_0
    T_curr = t                 # T_1
    
    result = np.zeros_like(x, dtype=np.float64)
    
    # Add T_0
    result += params[0] * T_prev
    
    # Add T_1
    result += params[1] * T_curr
    
    # Add T_2 to T_9
    for n in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        result += params[n] * T_next
        T_prev = T_curr
        T_curr = T_next
        
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
