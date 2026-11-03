import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 1.0 / (2 * N)  # 1/100
START_TIME = time.time()
TIME_LIMIT = 140.0

def _autoconv_peak_continuous(h):
    """Compute max of continuous autoconvolution of step function with heights h."""
    # Fine grid approach
    M = 8  # subdivisions per step
    h_fine = np.repeat(h, M)
    fine_delta = DELTA / M
    # Linear convolution
    conv = np.convolve(h_fine, h_fine, mode='full') * fine_delta ** 2
    return np.max(conv)

def _score(h):
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    ac_peak = _autoconv_peak_continuous(h)
    if ac_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * ac_peak)

def _objective(h):
    return -_score(h)

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return list(np.ones(N) / N)
    
    best_h = None
    best_score = -np.inf
    
    rng = np.random.default_rng(123)
    
    # Multiple initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Sparse random
    for k in [3, 5, 8, 12]:
        for trial in range(3):
            idx = rng.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0 / k
            inits.append(h)
    
    # Spectral design
    for _ in range(5):
        spec = np.zeros(N, dtype=complex)
        n_f = rng.integers(3, 15)
        freqs = rng.choice(N, size=n_f, replace=False)
        amps = rng.random(n_f)
        phases = rng.uniform(0, 2*np.pi, n_f)
        spec[freqs] = amps * np.exp(1j*phases)
        h = np.real(np.fft.ifft(spec))
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h = h / np.sum(h)
            inits.append(h)
    
    # Optimize
    bounds = [(0, None)] * N
    
    for h0 in inits:
        if time.time() - START_TIME > TIME_LIMIT - 3:
            break
        try:
            res = minimize(_objective, h0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            h_opt = np.maximum(res.x, 0)
            s = _score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except:
            pass
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    return list(best_h)
