import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Legendre polynomials P_0 to P_9
    # P_0(t) = 1
    # P_1(t) = t
    # (n+1)P_{n+1}(t) = (2n+1)tP_n(t) - nP_{n-1}(t)
    
    N = len(x)
    P = np.zeros((N, 10))
    P[:, 0] = 1.0
    P[:, 1] = t
    for k in range(2, 10):
        # k is the index in the array, corresponding to P_k
        # Recurrence for P_k uses P_{k-1} and P_{k-2}
        # P_k = ((2k-1)tP_{k-1} - (k-1)P_{k-2}) / k
        P[:, k] = ((2.0 * k - 1.0) * t * P[:, k-1] - (k - 1.0) * P[:, k-2]) / k
        
    # Linear combination
    return P @ params
