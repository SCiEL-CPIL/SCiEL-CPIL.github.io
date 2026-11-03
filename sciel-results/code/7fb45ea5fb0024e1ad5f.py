import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110

def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_score(h, tau=5.0):
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-9:
        return 0.0
    return m + tau * np.log(np.sum(np.exp((cs - m) / tau)))

def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _polish_time_domain(h0):
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max_score(x, tau=2.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
        if res.success and res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    return h0

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        if np.any(h > 1.0 + 1e-9) or np.any(h < -1e-9):
            return
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Fallback: Constant
    _update_best(np.full(N, 0.5))
    # Fallback: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)

    # Sparse Frequency Optimization
    def _h_from_x(x_params, support):
        x = np.zeros(N, dtype=np.complex128)
        if not support:
            return np.zeros(N)
        
        x[support[0]] = x_params[0]
        
        for i, f in enumerate(support[1:]):
            idx_re = 1 + 2*i
            idx_im = 2 + 2*i
            if idx_re >= len(x_params): break
            x[f] = x_params[idx_re] + 1j * x_params[idx_im]
            x[N-f] = np.conj(x[f])
            
        y = np.fft.ifft(x)
        h_raw = np.abs(y)**2
        sum_h = np.sum(h_raw)
        if sum_h == 0:
            return np.zeros(N)
        h = h_raw / sum_h * (N / 2.0)
        return h

    def _objective(params, support):
        h = _h_from_x(params, support)
        penalty = 0.0
        if np.any(h > 1.0):
            penalty += np.sum((h[h > 1.0] - 1.0)**2) * 100.0
        if np.any(h < 0.0):
            penalty += np.sum((-h[h < 0.0])**2) * 100.0
        
        s = _smooth_max_score(h, tau=5.0)
        return s + penalty

    # Try different supports
    supports = []
    # Low frequencies
    for K in [5, 8, 10, 12, 15]:
        supports.append(list(range(K)))
    # High frequencies
    for K in [5, 8, 10]:
        supports.append([0] + list(range(N-K+1, N)))
    # Bandpass
    supports.append([0] + list(range(2, 8)))
    supports.append([0] + list(range(10, 16)))
    supports.append([0] + list(range(35, 45)))
    # Random
    np.random.seed(123)
    for _ in range(10):
        K = np.random.randint(5, 15)
        idx = np.random.choice(np.arange(1, N//2), K, replace=False)
        supports.append([0] + list(idx))

    for support in supports:
        if time.time() - t0 > _DEADLINE - 30:
            break
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Initial guess: 
        # We want to approximate a cosine in time domain.
        # A cosine in time domain is two spikes in frequency domain at +/- f0.
        # So we initialize one pair of frequencies with high amplitude and others low.
        
        x0 = np.zeros(num_params)
        x0[0] = 1.0 # DC component
        
        # Choose the first non-DC frequency to be the "main" carrier
        # support[1] is the first non-DC freq
        if K > 1:
            main_idx = 1
            # Set amplitude for the main frequency pair
            # In _h_from_x, index 1 is Re, 2 is Im for support[1]
            x0[1] = 0.5 # Re
            x0[2] = 0.0 # Im
            
            # Add some small random perturbations to other frequencies to help escape symmetry
            for i in range(2, K-1):
                idx_re = 1 + 2*i
                idx_im = 2 + 2*i
                if idx_re >= len(x_params): break
                x0[idx_re] = np.random.rand() * 0.05
                x0[idx_im] = np.random.rand() * 0.05

        # Stage 1: Global Search (Nelder-Mead) to find a good basin
        # We use a simplified objective that is faster if possible, but here we stick to the same one.
        # Nelder-Mead can be slow in high dims, so we limit iterations.
        try:
            res_nm = minimize(_objective, x0, args=(support,), method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            x_start = res_nm.x
        except Exception:
            x_start = x0

        # Stage 2: Local Refinement (L-BFGS-B) for precision
        try:
            res_lbfgs = minimize(_objective, x_start, args=(support,), method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_lbfgs.fun < best_score:
                h_opt = _h_from_x(res_lbfgs.x, support)
                _update_best(h_opt)
        except Exception:
            pass

    # Polish the best result in time domain
    if best_h is not None:
        polished_h = _polish_time_domain(best_h)
        if _score(polished_h) < best_score:
            best_h = polished_h
            best_score = _score(polished_h)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_h = _polish_time_domain(best_h)
        
    return best_h
