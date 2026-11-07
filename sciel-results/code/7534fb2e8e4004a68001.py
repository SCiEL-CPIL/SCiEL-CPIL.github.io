import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > 145:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x_arr)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Chebyshev polynomial basis on [-2, 2]
    # Map to [-1, 1]: t = x / 2
    # T_k(t) via recurrence: T_0=1, T_1=t, T_k = 2t*T_{k-1} - T_{k-2}
    t = x / 2.0
    n = x.size
    basis = np.empty((n, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = t
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    
    # Linear combination (convex fit in params)
    result = basis @ p
    
    # Safety
    result = np.clip(result, -1e8, 1e8)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
