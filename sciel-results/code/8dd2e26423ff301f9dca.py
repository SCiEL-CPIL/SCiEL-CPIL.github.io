import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N  # interval width
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _autoconv_max_smooth(heights, beta=200.0):
    """Compute smooth max of normalized autoconvolution."""
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    # Normalize to unit integral
    h_norm = h / integral
    # Linear autoconvolution via FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    # Smooth max via log-sum-exp
    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0
    shifted = conv - mx
    return np.log(np.sum(np.exp(beta * shifted))) / beta + mx

def _objective(x):
    return _autoconv_max_smooth(x)

def construct():
    global START_TIME
    best_x = np.ones(N)
    best_val = _objective(best_x)

    # Phase 1: L-BFGS-B from uniform
    try:
        res = minimize(_objective, best_x.copy(), method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100})
        if res.fun < best_val:
            best_x = np.maximum(res.x, 0.0)
            best_val = res.fun
    except Exception:
        pass

    if time.time() - START_TIME > TIME_LIMIT - 30:
        return best_x.tolist()

    # Phase 2: Try different initializations
    inits = [
        np.ones(N),
        np.exp(-4.0 * (np.linspace(-1, 1, N)) ** 2),
        np.cos(np.pi * np.linspace(-1, 1, N)) ** 2,
    ]
    for x0 in inits:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val:
                best_x = np.maximum(res.x, 0.0)
                best_val = res.fun
        except Exception:
            pass

    # Phase 3: High-precision polish
    if time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            res = minimize(_objective, best_x.copy(), method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200})
            if res.fun < best_val:
                best_x = np.maximum(res.x, 0.0)
                best_val = res.fun
        except Exception:
            pass

    return best_x.tolist()
