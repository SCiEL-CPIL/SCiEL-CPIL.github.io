import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Legendre polynomial basis of degree 9.
    Legendre polynomials are orthogonal with respect to the uniform distribution
    on [-1, 1], which matches the uniform input distribution on [-3, 3] after scaling.
    
    Parameters:
    x (np.ndarray): Input array of shape (N,), values in [-3, 3].
    params (np.ndarray): Array of 10 coefficients to be fitted.
    
    Returns:
    np.ndarray: Predictions of shape (N,).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    # Map x from [-3, 3] to [-1, 1] for Legendre polynomials
    t = x / 3.0
    
    if len(t) == 0:
        return np.zeros_like(t)
        
    # Precompute Legendre polynomials P_0 to P_9
    # P_0(t) = 1
    # P_1(t) = t
    # P_{n+1}(t) = ((2n+1)t*P_n(t) - n*P_{n-1}(t)) / (n+1)
    
    result = np.zeros_like(t)
    
    # P_0
    P_nm1 = np.ones_like(t)
    result += params[0] * P_nm1
    
    # P_1
    P_n = t
    result += params[1] * P_n
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    # Compute P_2 to P_9
    for k in range(2, 10):
        # n is the previous degree (k-1)
        n = k - 1
        P_np1 = ((2.0 * n + 1.0) * t * P_n - n * P_nm1) / (n + 1.0)
        result += params[k] * P_np1
        P_nm1 = P_n
        P_n = P_np1
        
        # Check deadline periodically
        if time.time() > deadline:
            break
            
    return result
