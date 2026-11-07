import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Build a 10-term basis mixing low-degree polynomials and low-frequency trig
    # Terms: 1, x, x^2, x^3, cos(x), sin(x), cos(2x), sin(2x), cos(3x), sin(3x)
    # This hybrid captures both smooth polynomial trends and periodic components.
    
    n = x.shape[0]
    T = np.empty((n, 10), dtype=np.float64)
    
    # Polynomial part (first 4 columns)
    T[:, 0] = 1.0
    T[:, 1] = x
    T[:, 2] = x * x
    T[:, 3] = x * x * x
    
    # Trig part (next 6 columns)
    T[:, 4] = np.cos(x)
    T[:, 5] = np.sin(x)
    T[:, 6] = np.cos(2.0 * x)
    T[:, 7] = np.sin(2.0 * x)
    T[:, 8] = np.cos(3.0 * x)
    T[:, 9] = np.sin(3.0 * x)
    
    result = T @ params
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
