import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 1 input.
    Uses a Chebyshev series expansion of degree 9 on the interval [-2, 2].
    This provides a better-conditioned basis for approximating smooth functions
    compared to monomials, utilizing all 10 parameters.
    """
    # Deadline guard
    if time.time() - _START_TIME > 145.0:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Map x from [-2, 2] to t in [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.empty((10, x.size), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    
    for n in range(2, 10):
        T[n] = 2.0 * t * T[n-1] - T[n-2]
    
    # Linear combination: y = sum(p[i] * T[i])
    # Efficient dot product: (10, N) dot (10,) -> (N,)
    # Or (N, 10) dot (10,)
    # T is (10, N). p is (10,).
    # result = np.sum(p[:, np.newaxis] * T, axis=0)
    # Or simply:
    result = T[0] * p[0] + T[1] * p[1] + T[2] * p[2] + T[3] * p[3] + T[4] * p[4] + \
             T[5] * p[5] + T[6] * p[6] + T[7] * p[7] + T[8] * p[8] + T[9] * p[9]
             
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
