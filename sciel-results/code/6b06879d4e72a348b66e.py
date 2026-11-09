import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis of Fourier series and polynomials.
    
    The basis is designed to be well-conditioned for BFGS optimization starting from ones.
    Input x is in [-3, 3]. We scale to t in [-1, 1].
    
    Parameters:
    p[0]: DC offset
    p[1]: Amp sin(pi t)
    p[2]: Amp cos(pi t)
    p[3]: Amp sin(2 pi t)
    p[4]: Amp cos(2 pi t)
    p[5]: Amp sin(3 pi t)
    p[6]: Amp cos(3 pi t)
    p[7]: Amp sin(4 pi t)
    p[8]: Amp cos(4 pi t)
    p[9]: Quartic coeff (t^4)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Scale input to [-1, 1]
    t = x / 3.0
    
    # Pre-compute basis functions
    # Trig terms - extended to frequency 4 for better coverage
    sin_1 = np.sin(np.pi * t)
    cos_1 = np.cos(np.pi * t)
    sin_2 = np.sin(2.0 * np.pi * t)
    cos_2 = np.cos(2.0 * np.pi * t)
    sin_3 = np.sin(3.0 * np.pi * t)
    cos_3 = np.cos(3.0 * np.pi * t)
    sin_4 = np.sin(4.0 * np.pi * t)
    cos_4 = np.cos(4.0 * np.pi * t)
    
    # Polynomial term - quartic for capturing smooth non-periodic trends
    t_4 = t * t * t * t
    
    # Combine linearly
    y = np.full_like(t, params[0], dtype=float)
    y += params[1] * sin_1
    y += params[2] * cos_1
    y += params[3] * sin_2
    y += params[4] * cos_2
    y += params[5] * sin_3
    y += params[6] * cos_3
    y += params[7] * sin_4
    y += params[8] * cos_4
    y += params[9] * t_4
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return y
