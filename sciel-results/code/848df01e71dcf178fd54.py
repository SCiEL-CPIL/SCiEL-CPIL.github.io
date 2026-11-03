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

def _polish_time_domain(h0, max_iter=300, ftol=1e-12):
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max_score(x, tau=1.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': ftol})
        if res.success and res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    
    # Try with different tau values
    for tau in [0.5, 2.0, 3.0]:
        def obj_t(x):
            return _smooth_max_score(x, tau=tau)
        try:
            res = minimize(obj_t, h0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-10})
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
        
        s = _smooth_max_score(h, tau=3.0)
        return s + penalty

    # Generate diverse supports based on spectral analysis
    supports = []
    
    # Low frequency bands
    for K in [8, 12, 16, 20, 25]:
        supports.append(list(range(K)))
    
    # Mixed low-mid frequencies
    for K in [10, 15, 20]:
        supports.append([0] + list(range(1, 5)) + list(range(N//4, N//4 + K-5)))
    
    # Random supports with structure
    np.random.seed(123)
    for trial in range(20):
        K = np.random.randint(8, 22)
        # Ensure DC is included
        idx = np.random.choice(np.arange(1, N//2), K-1, replace=False)
        idx_sorted = sorted(idx)
        supports.append([0] + list(idx_sorted))
    
    # Supports based on cosine-like patterns
    for freq in [1, 2, 3]:
        K = min(N//2, freq * 5 + 1)
        supports.append(list(range(K)))
    
    # Optimize each support with multiple random initializations
    for support in supports:
        if time.time() - t0 > _DEADLINE - 30:
            break
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Try multiple random initializations
        n_starts = 3 if K <= 15 else 2
        for start in range(n_starts):
            if time.time() - t0 > _DEADLINE - 30:
                break
            
            x0 = np.zeros(num_params)
            x0[0] = 1.0  # DC component
            
            if start == 0:
                # Small random phases
                for i in range(K-1):
                    x0[1+2*i] = 0.05 * np.cos(np.random.rand()*2*np.pi)
                    x0[2+2*i] = 0.05 * np.sin(np.random.rand()*2*np.pi)
            elif start == 1:
                # Larger random amplitudes
                for i in range(K-1):
                    amp = 0.1 * np.random.rand()
                    x0[1+2*i] = amp * np.cos(np.random.rand()*2*np.pi)
                    x0[2+2*i] = amp * np.sin(np.random.rand()*2*np.pi)
            else:
                # Structured initialization
                for i in range(K-1):
                    f = support[i+1]
                    phase = 2.0 * np.pi * f / N
                    x0[1+2*i] = 0.08 * np.cos(phase)
                    x0[2+2*i] = 0.08 * np.sin(phase)
            
            try:
                res = minimize(_objective, x0, args=(support,), method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-10})
                
                if res.fun < best_score:
                    h_opt = _h_from_x(res.x, support)
                    _update_best(h_opt)
            except Exception:
                pass
    
    # Additional refinement: try to improve best solution in time domain
    if best_h is not None:
        # Multiple polishing attempts with different parameters
        for attempt in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            
            if attempt == 0:
                polished_h = _polish_time_domain(best_h, max_iter=200, ftol=1e-10)
            elif attempt == 1:
                # Perturb slightly before polishing
                perturbed = best_h + 0.01 * np.random.randn(N)
                perturbed = _fix_sum(perturbed)
                polished_h = _polish_time_domain(perturbed, max_iter=150, ftol=1e-10)
            else:
                polished_h = _polish_time_domain(best_h, max_iter=100, ftol=1e-12)
            
            s_polished = _score(polished_h)
            if s_polished < best_score:
                best_h = polished_h
                best_score = s_polished
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_h = _polish_time_domain(best_h)
        
    return best_h
