import numpy as np
import time

_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Structural Change: Chebyshev Polynomial Series
    # Map x from [-2, 2] to t in [-1, 1]
    # t = x / 2
    # y(x) = sum_{i=0}^{9} p_i * T_i(t)
    # T_i(t) are Chebyshev polynomials of the first kind.
    # This model is linear in parameters, making the optimization landscape convex
    # (MSE is a quadratic bowl in p). BFGS should converge rapidly and reliably.
    
    t = x / 2.0
    
    # Compute Chebyshev polynomials up to degree 9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, x.shape[0]), dtype=np.float64)
    T[0, :] = 1.0
    if x.shape[0] > 0:
        T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
        
    # y = T @ p
    y = T @ p
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
