import numpy as np
import time
from scipy.optimize import minimize

N = 600
_x = np.linspace(-0.25, 0.25, N + 1)
_xc = 0.5 * (_x[:-1] + _x[1:])
_dx = _x[1] - _x[0]

def _autoconv_max_ratio(h):
    """Compute the normalized maximum of the autoconvolution."""
    # h is nonnegative step heights on N equal intervals
    # Autoconvolution: (h*h)(k) = sum_i h_i * h_{i+k} (discrete)
    # We need the continuous version: (h*h)(t) = integral h(s)h(t-s)ds
    # For step functions, we can compute via FFT or direct convolution
    # Normalize: max(h*h) / (sum(h_i)*dx)^2
    total = np.sum(h) * _dx
    if total <= 0:
        return np.inf
    # Compute autoconvolution using FFT
    n_fft = 2 * N
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2, n_fft)[:N] * _dx
    peak = np.max(conv)
    return peak / (total ** 2)

def _params_to_heights(params):
    """Convert Gaussian mixture parameters to step heights."""
    # params: for each component: (center, width, amplitude)
    # K components
    K = len(params) // 3
    h = np.zeros(N)
    for k in range(K):
        c = params[3*k]
        w = params[3*k + 1]
        a = params[3*k + 2]
        if w <= 0:
            w = 1e-6
        # Gaussian: a * exp(-0.5*((x-c)/w)^2)
        g = a * np.exp(-0.5 * ((_xc - c) / w) ** 2)
        h += g
    return np.maximum(h, 0.0)

def _objective(params):
    h = _params_to_heights(params)
    return _autoconv_max_ratio(h)

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    best_score = _autoconv_max_ratio(np.ones(N))
    best_params = None
    
    # Start with a few Gaussian components
    K = 6  # number of Gaussian components
    # Initial guess: symmetric Gaussians
    centers = np.linspace(-0.15, 0.15, K)
    widths = np.full(K, 0.05)
    amplitudes = np.full(K, 1.0)
    init_params = np.concatenate([centers, widths, amplitudes])
    
    # Try several random initializations
    np.random.seed(42)
    for trial in range(8):
        if time.time() > deadline:
            break
        
        if trial == 0:
            params0 = init_params.copy()
        else:
            centers_r = np.random.uniform(-0.2, 0.2, K)
            widths_r = np.random.uniform(0.02, 0.15, K)
            amplitudes_r = np.random.uniform(0.5, 2.0, K)
            params0 = np.concatenate([centers_r, widths_r, amplitudes_r])
        
        try:
            res = minimize(_objective, params0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is not None:
        h = _params_to_heights(best_params)
        # Ensure nonnegative
        h = np.maximum(h, 0.0)
        return list(h)
    else:
        return [1.0] * N
