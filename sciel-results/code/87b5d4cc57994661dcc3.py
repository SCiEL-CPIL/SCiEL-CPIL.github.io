import time
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)

    t = (x - 3.05) / 2.95
    
    # Check time
    if time.time() > deadline:
        return np.zeros_like(x, dtype=np.float64)

    # Chebyshev Polynomials
    phi0 = 1.0
    phi1 = t
    phi2 = 2.0 * t**2 - 1.0
    phi3 = 4.0 * t**3 - 3.0 * t
    phi4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # RBFs
    lam = 10.0
    centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    phi5 = np.exp(-lam * (t - centers[0])**2)
    phi6 = np.exp(-lam * (t - centers[1])**2)
    phi7 = np.exp(-lam * (t - centers[2])**2)
    phi8 = np.exp(-lam * (t - centers[3])**2)
    phi9 = np.exp(-lam * (t - centers[4])**2)
    
    result = (p[0] * phi0 +
              p[1] * phi1 +
              p[2] * phi2 +
              p[3] * phi3 +
              p[4] * phi4 +
              p[5] * phi5 +
              p[6] * phi6 +
              p[7] * phi7 +
              p[8] * phi8 +
              p[9] * phi9)
              
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x, dtype=np.float64)
        
    return result
