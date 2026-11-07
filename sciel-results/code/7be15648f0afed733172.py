import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a mixed basis of Chebyshev polynomials and
    trigonometric functions, designed to capture both polynomial and periodic
    components of the target function on [-2, 2].
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4 (5 terms)
    T = np.zeros((5, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    for n in range(2, 5):
        T[n] = 2.0 * t * T[n-1] - T[n-2]
    
    # Compute trigonometric basis (5 terms)
    # sin(x), cos(x), sin(2x), cos(2x), sin(3x)
    trig = np.zeros((5, x.shape[0]), dtype=np.float64)
    trig[0] = np.sin(x)
    trig[1] = np.cos(x)
    trig[2] = np.sin(2.0 * x)
    trig[3] = np.cos(2.0 * x)
    trig[4] = np.sin(3.0 * x)
    
    # Combine: first 5 params for Chebyshev, next 5 for trig
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(5):
        result += params[i] * T[i]
        result += params[i + 5] * trig[i]
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
