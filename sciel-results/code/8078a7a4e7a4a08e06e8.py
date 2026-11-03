import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin from 150s

def _autoconv_linear(h):
    """Compute linear autoconvolution of h."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _max_abs_ac_offpeak(ac):
    """Max absolute autoconvolution excluding k=0 (index N-1)."""
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    return np.max(np.abs(ac_no_peak))

def _score(h):
    """Compute the ratio: max |C(k)| for k!=0 divided by (sum h)^2."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return np.inf
    ac = _autoconv_linear(h)
    max_abs = _max_abs_ac_offpeak(ac)
    return max_abs / (S * S)

def _smooth_obj(h, beta):
    """Smooth objective for gradient-based optimization."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return 1e10
    ac = _autoconv_linear(h)
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    abs_ac = np.abs(ac_no_peak)
    if beta < 1e6:
        log_sum = np.logaddexp.reduce(np.log(abs_ac + 1e-30) * beta) / beta
    else:
        log_sum = np.max(abs_ac)
    return log_sum / (S * S)

def _local_search_flip(h, max_iters=2000):
    """Sign-flip local search to minimize max |C(k)| for k!=0."""
    h = h.copy().astype(float)
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    
    best_score = _score(h)
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            h[i] = -h[i]
            S = np.sum(h)
            if abs(S) < 1e-15:
                h[i] = -h[i]
                continue
            ac = _autoconv_linear(h)
            max_abs = _max_abs_ac_offpeak(ac)
            new_score = max_abs / (S * S)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h

def _local_search_block(h, max_iters=500):
    """Block sign-flip local search."""
    h = h.copy().astype(float)
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    
    best_score = _score(h)
    block_sizes = [1, 2, 4, 8, 16]
    
    for it in range(max_iters):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, bs):
                h_old = h[start:start+bs].copy()
                h[start:start+bs] = -h_old
                S = np.sum(h)
                if abs(S) < 1e-15:
                    h[start:start+bs] = h_old
                    continue
                ac = _autoconv_linear(h)
                max_abs = _max_abs_ac_offpeak(ac)
                new_score = max_abs / (S * S)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    h[start:start+bs] = h_old
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Try diverse initializations with L-BFGS optimization
    inits = []
    
    # All ones
    inits.append(np.ones(N))
    
    # Top-hat of various widths
    for w in [50, 100, 150, 200]:
        h = np.zeros(N)
        h[:w] = 1.0
        inits.append(h)
    
    # Gaussian-like
    t = np.arange(N)
    for sigma in [50, 100, 150]:
        h = np.exp(-0.5 * ((t - N//2) / sigma) ** 2)
        inits.append(h)
    
    # Cosine window
    t = np.arange(N)
    h = np.cos(np.pi * t / N) ** 2
    inits.append(h)
    
    # Random smooth
    for _ in range(5):
        h = np.random.randn(N)
        h = np.convolve(h, np.ones(10)/10, mode='same')
        inits.append(h)
    
    # Random with bias
    for _ in range(3):
        h = np.random.uniform(-1, 1, size=N)
        h += 0.3
        inits.append(h)
    
    for h0 in inits:
        if elapsed() > DEADLINE - 30:
            break
        
        S = np.sum(h0)
        if abs(S) < 1e-15:
            h0 = h0 + 1.0
            S = np.sum(h0)
        h0 = h0 / S
        
        # L-BFGS optimization with smooth surrogate
        beta = 50.0
        try:
            res = minimize(
                _smooth_obj, h0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        S = np.sum(h_opt)
        if abs(S) < 1e-15:
            h_opt = h_opt + 1.0
            S = np.sum(h_opt)
        h_opt = h_opt / S
        
        sc = _score(h_opt)
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # Polish with local search
        if elapsed() > DEADLINE - 10:
            break
        
        h_polished = _local_search_block(h_opt, max_iters=200)
        sc = _score(h_polished)
        if sc < best_score:
            best_score = sc
            best_h = h_polished.copy()
    
    # Phase 2: More aggressive local search on best
    if elapsed() < DEADLINE - 5:
        h_final = _local_search_flip(best_h, max_iters=3000)
        sc = _score(h_final)
        if sc < best_score:
            best_score = sc
            best_h = h_final.copy()
    
    return best_h
