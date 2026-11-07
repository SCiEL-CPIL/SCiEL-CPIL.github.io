import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a mixed basis: 3 Chebyshev polynomial terms
    + 7 adaptive tanh hidden units with fixed centers/scales.
    
    The Chebyshev backbone (T0..T2 on t=x/3) provides a stable global
    polynomial approximation. The 7 tanh units, each with a fixed center
    spread across [-3, 3] and a fixed scale, provide localized nonlinear
    flexibility. BFGS jointly optimizes all 10 parameters.
    
    params (length 10):
      params[0..2]: Chebyshev coefficients (T0, T1, T2)
      params[3..9]: tanh unit weights
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Map x to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Chebyshev polynomials T0, T1, T2
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    
    # 7 tanh units with fixed centers and scales
    # Centers spread across the domain [-3, 3]
    centers = np.array([-2.8, -1.8, -0.9, 0.0, 0.9, 1.8, 2.8])
    # Scales: larger at the edges (wider transition), smaller in the middle
    scales = np.array([0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.8])
    
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2)
    
    for i in range(7):
        y += params[3 + i] * np.tanh(scales[i] * (x - centers[i]))
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
