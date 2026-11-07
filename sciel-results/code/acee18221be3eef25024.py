import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a degree-9 Chebyshev polynomial in x.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, coefficients for T_0 through T_9
    
    Returns: array of predictions, one per point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    if n == 0:
        return np.empty(0, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev basis
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Build Chebyshev basis via Clenshaw recurrence (no full matrix needed)
    # result = sum_{k=0}^{9} params[k] * T_k(t)
    # Use Clenshaw's algorithm for numerical stability
    b_kp2 = 0.0  # b_{k+2}
    b_kp1 = 0.0  # b_{k+1}
    
    # Start from the highest degree (k=9) down to 0
    for k in range(9, -1, -1):
        b_k = 2.0 * t * b_kp1 - b_kp2 + params[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    
    return b_kp2
