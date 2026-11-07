import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Chebyshev Polynomial Basis.
    
    Strategy:
    - Map input domain [0.1, 6] to [-1, 1] to utilize the orthogonality of Chebyshev polynomials.
    - Use Chebyshev polynomials of the first kind T_0(t) to T_9(t) as the basis functions.
    - This results in a linear-in-parameters model with a well-conditioned design matrix.
    - BFGS will converge rapidly to the global minimum because the loss surface is convex 
      (quadratic in parameters) and the basis is numerically stable.
    
    Parameters (10 total):
    p[0] to p[9]: Coefficients for T_0(t) to T_9(t).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)

    # Map x from [0.1, 6] to t in [-1, 1]
    # t = 2 * (x - min_x) / (max_x - min_x) - 1
    # min_x = 0.1, max_x = 6.0
    # range = 5.9
    # t = 2 * (x - 0.1) / 5.9 - 1
    min_x = 0.1
    max_x = 6.0
    t = 2.0 * (x - min_x) / (max_x - min_x) - 1.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
        
    # The model is linear in parameters: y = sum(p[i] * T[i](t))
    # T is shape (10, N), p is shape (10,)
    # result = T.T @ p
    result = T.T @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
