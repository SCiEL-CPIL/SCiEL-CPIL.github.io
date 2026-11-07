"""Symbolic regression with a well-conditioned Chebyshev polynomial basis."""
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using 10 parameters.
    
    Basis design:
    - x is in [0.1, 6]. We map to t = (x - 3.05) / 2.95 -> [-1, 1]
    - We use Chebyshev polynomials of the first kind T_0 through T_9.
      These are near-orthogonal on [-1, 1] and provide excellent conditioning
      for polynomial approximation of smooth functions.
    
    Basis functions (10 total):
    b0 = T_0(t) = 1
    b1 = T_1(t) = t
    b2 = T_2(t) = 2t^2 - 1
    b3 = T_3(t) = 4t^3 - 3t
    ... up to T_9
    
    This gives BFGS a smooth, well-conditioned landscape for polynomial targets.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    t = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_{n+1}(t) = 2t*T_n(t) - T_{n-1}(t)
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination: pred = sum(params[i] * T[i])
    pred = np.sum(params[:, np.newaxis] * T, axis=0)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
