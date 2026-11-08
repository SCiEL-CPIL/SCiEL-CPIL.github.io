"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution via FFT
    n_conv = 2 * N - 1
    h_fft = np.fft.rfft(h, n=n_conv)
    autoconv = np.fft.irfft(h_fft * np.conj(h_fft), n=n_conv)
    linf = np.max(np.abs(autoconv))
    if linf == 0:
        return 0.0
    return l2_sq / (l1 * linf)

def _neg_ratio(params):
    """Negative ratio for minimization"""
    h = np.clip(params, 0, None)
    return -_ratio(h)

def construct():
    deadline = time.time() + 120.0
    best_h = None
    best_val = 0.0
    
    # Try several initial configurations
    configs = []
    
    # 2-point adjacent cluster with 2:1 ratio
    h2 = np.zeros(N)
    h2[0] = 2.0/3.0
    h2[1] = 1.0/3.0
    configs.append(h2)
    
    # 3-point cluster
    h3 = np.zeros(N)
    h3[0] = 1.0/6.0
    h3[1] = 4.0/6.0
    h3[2] = 1.0/6.0
    configs.append(h3)
    
    # Uniform
    huni = np.ones(N) / N
    configs.append(huni)
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        hrand = rng.random(N)
        hrand /= np.sum(hrand)
        configs.append(hrand)
    
    # Evaluate initial configs
    for h in configs:
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Optimize from best initial config
    if best_h is not None:
        try:
            res = minimize(_neg_ratio, best_h.copy(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            h_opt = np.clip(res.x, 0, None)
            val_opt = _ratio(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try more random restarts if time permits
    start_time = time.time()
    for i in range(20):
        if time.time() > deadline - 30:
            break
        if (time.time() - start_time) > 100:
            break
        hrand = rng.random(N)
        hrand /= np.sum(hrand)
        try:
            res = minimize(_neg_ratio, hrand.copy(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
            h_opt = np.clip(res.x, 0, None)
            val_opt = _ratio(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    return list(best_h)
# EVOLVE-BLOCK-END
