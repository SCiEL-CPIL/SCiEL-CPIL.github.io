import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

def construct():
    deadline = time.time() + 140.0
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Single spike initialization
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    spike_score = _compute_ratio(spike)
    if spike_score > best_score:
        best_score = spike_score
        best_h = spike.copy()
    
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            h0 = best_h.copy()
            h0 = np.maximum(h0, 1e-8)
        elif r == 1:
            h0 = np.zeros(N)
            h0[N // 2] = 1.0
            h0[N // 2 - 1] = 0.5
        elif r == 2:
            h0 = np.zeros(N)
            h0[N // 2] = 1.0
            h0[N // 2 + 1] = 0.5
        elif r == 3:
            h0 = np.zeros(N)
            h0[N // 2] = 1.0
            h0[N // 2 - 2] = 0.3
            h0[N // 2 + 2] = 0.3
        elif r == 4:
            h0 = np.zeros(N)
            for i in range(max(0, N // 2 - 4), min(N, N // 2 + 5)):
                h0[i] = np.exp(-((i - N // 2) ** 2) / 4.0)
        elif r == 5:
            h0 = np.zeros(N)
            h0[N // 2] = 1.0
            h0[N // 2 - 1] = 0.8
            h0[N // 2 + 1] = 0.8
        else:
            h0 = best_h * (1.0 + 0.2 * np.random.randn(N))
            h0 = np.maximum(h0, 1e-8)
        
        log_h0 = np.log(h0)
        
        try:
            result = minimize(
                _neg_ratio,
                log_h0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.exp(result.x)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    return [float(x) for x in best_h]
