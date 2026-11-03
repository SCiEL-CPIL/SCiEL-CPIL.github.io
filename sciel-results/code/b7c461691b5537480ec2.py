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
        # x_params: [Re(x0), Im(x1), Re(x1), Im(x2), ...]
        # support: list of indices
        # x0 is real, others complex
        x = np.zeros(N, dtype=np.complex128)
        if not support:
            return np.zeros(N)
        
        # x0 is real
        x[support[0]] = x_params[0]
        
        # Process pairs
        # The support list should be ordered: [f0, f1, ..., fK-1]
        # We assume f0=0.
        # For k > 0, x_k is complex, x_{N-k} is conjugate.
        # We optimize x_k.
        
        # Mapping params to x
        # Param 0: Re(x_0)
        # Param 1: Re(x_1), Param 2: Im(x_1)
        # Param 3: Re(x_2), Param 4: Im(x_2)
        
        # Let's just map directly based on support length
        # support[0] is 0.
        # support[1:] are the positive frequencies.
        
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
        # Penalty for bounds
        penalty = 0.0
        if np.any(h > 1.0):
            penalty += np.sum((h[h > 1.0] - 1.0)**2) * 100.0
        if np.any(h < 0.0):
            penalty += np.sum((-h[h < 0.0])**2) * 100.0
        
        s = _score(h)
        return s + penalty

    # Try different supports
    supports = []
    # Low frequencies
    for K in [10, 20, 30]:
        supports.append(list(range(K)))
    # High frequencies
    for K in [10, 20]:
        supports.append([0] + list(range(N-K+1, N)))
    # Random
    for _ in range(5):
        K = 20
        idx = np.random.choice(np.arange(1, N//2), K, replace=False)
        supports.append([0] + list(idx))

    for support in supports:
        if time.time() - t0 > _DEADLINE - 10:
            break
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Initial guess: random phases, small amplitudes
        x0 = np.zeros(num_params)
        x0[0] = 1.0 # DC component
        # Random phases for others
        for i in range(K-1):
            x0[1+2*i] = np.cos(np.random.rand()*2*np.pi) * 0.1
            x0[2+2*i] = np.sin(np.random.rand()*2*np.pi) * 0.1
            
        res = minimize(_objective, x0, args=(support,), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_score:
            h_opt = _h_from_x(res.x, support)
            _update_best(h_opt)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
