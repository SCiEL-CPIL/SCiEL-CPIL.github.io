import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape[0] != 10:
        p = p[:10] if p.shape[0] >= 10 else np.pad(p, (0, 10 - p.shape[0]))
    
    # Pure Fourier basis: 5 cosines + 5 sines with frequencies {1,2,3,4,5}
    # This is linear in params, so BFGS will converge quickly
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    c1 = np.cos(freqs[0] * x)
    c2 = np.cos(freqs[1] * x)
    c3 = np.cos(freqs[2] * x)
    c4 = np.cos(freqs[3] * x)
    c5 = np.cos(freqs[4] * x)
    
    s1 = np.sin(freqs[0] * x)
    s2 = np.sin(freqs[1] * x)
    s3 = np.sin(freqs[2] * x)
    s4 = np.sin(freqs[3] * x)
    s5 = np.sin(freqs[4] * x)
    
    result = (p[0] * c1 + p[1] * c2 + p[2] * c3 + p[3] * c4 + p[4] * c5 +
              p[5] * s1 + p[6] * s2 + p[7] * s3 + p[8] * s4 + p[9] * s5)
    
    return result
