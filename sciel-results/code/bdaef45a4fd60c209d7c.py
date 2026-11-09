import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2x - (x_max + x_min)) / (x_max - x_min)
    x_min = 0.1
    x_max = 6.0
    t = (2 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0 to T_9
    T = np.zeros((10, len(x)))
    T[0] = 1.0
    if len(x) > 0:
        T[1] = t
        for k in range(2, 10):
            T[k] = 2 * t * T[k-1] - T[k-2]
            
    # Linear combination
    # params is length 10
    # prediction = sum(params[k] * T[k])
    prediction = np.sum(params[:, np.newaxis] * T, axis=0)
    
    return prediction
