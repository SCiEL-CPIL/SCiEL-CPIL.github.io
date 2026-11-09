import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [0.1, 6] to [-1, 1]
    u = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T0 to T9
    T = np.empty((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = u
    
    for i in range(2, 10):
        T[i, :] = 2.0 * u * T[i-1, :] - T[i-2, :]
    
    # Linear combination: y = sum(params[i] * T[i])
    y = T @ params
    
    return y
