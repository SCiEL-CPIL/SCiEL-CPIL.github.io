import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def score(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * max_conv)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Candidate 1: single spike
    h_sp = np.zeros(N); h_sp[0] = 1.0
    s = score(h_sp)
    if s > best_score:
        best_score = s; best_h = h_sp.copy()
    
    # Candidate 2: [1, 0, 0, ...] variants with small neighbors
    for k in range(1, 6):
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            h_t = np.zeros(N)
            h_t[0] = 1.0 - frac
            if frac > 0:
                h_t[k] = frac
            s = score(h_t)
            if s > best_score:
                best_score = s; best_h = h_t.copy()
    
    # Water-filling initialization: start with uniform, then iteratively
    # reduce heights where local autoconvolution contribution is high
    def water_filling_init(support_size, n_iters=50):
        h = np.zeros(N)
        # Start with uniform on a small support
        support = np.random.choice(N, support_size, replace=False)
        h[support] = 1.0 / support_size
        
        for _ in range(n_iters):
            if time.time() > deadline - 10:
                break
            conv = np.convolve(h, h, mode='full')
            # Compute contribution of each point to peak autoconv
            # For point i, its contribution to autoconv[j] involves h[i]*h[j-i]
            # Simplify: reduce h where it correlates with itself at small lags
            l1 = np.sum(h)
            if l1 < 1e-15:
                break
            h_norm = h / l1
            conv_n = np.convolve(h_norm, h_norm, mode='full')
            # Gradient-like: reduce h where local density of autoconv peaks
            # Use the autoconvolution centered values as a proxy
            center = len(conv_n) // 2
            local_peak = conv_n[max(0, center - 5):center + 6]
            # Allocate inversely to local spectral density
            h_new = np.zeros(N)
            for i in range(N):
                # Estimate local interference
                if i > 0:
                    inter = sum(h_norm[max(0, i-d):i+1] * h_norm[max(0, i-d):i+1] for d in range(min(i, 4)))
                else:
                    inter = h_norm[0]**2
                h_new[i] = 1.0 / (1.0 + inter)
            # Blend
            h = 0.7 * h_new + 0.3 * h
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                h = h / np.sum(h)
        return h
    
    # Try water-filling with different support sizes
    np.random.seed(123)
    for sz in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            h_wf = water_filling_init(sz, n_iters=30)
            s = score(h_wf)
            if s > best_score:
                best_score = s; best_h = h_wf.copy()
    
    # Nelder-Mead refinement on the best candidate
    # Work on a reduced dimension (first 15 elements) to keep it fast
    def objective(x):
        h_full = np.zeros(N)
        h_full[:len(x)] = np.abs(x)
        return -score(h_full)
    
    # Multiple restarts from different initializations
    initializations = [best_h.copy()]
    
    # Add sparse random initializations
    for _ in range(8):
        if time.time() > deadline - 20:
            break
        sz = np.random.randint(2, 8)
        idx = np.random.choice(N, sz, replace=False)
        h_r = np.zeros(N)
        h_r[idx] = np.random.rand(sz)
        h_r = h_r / np.sum(h_r)
        initializations.append(h_r)
    
    for h_init in initializations:
        if time.time() > deadline - 5:
            break
        # Truncate to first 20 dimensions for optimization
        dim = min(20, N)
        x0 = h_init[:dim].copy()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            h_opt = np.zeros(N)
            h_opt[:dim] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: try a few more targeted sparse patterns
    if time.time() < deadline - 5:
        # Pattern: [a, b, 0, ..., c, 0, ...] with c far from a,b
        for a in np.linspace(0.3, 0.8, 10):
            for b in np.linspace(0.05, 0.4, 8):
                for c_pos in [10, 20, 30, 40]:
                    c = 1.0 - a - b
                    if c < 0:
                        continue
                    h_t = np.zeros(N)
                    h_t[0] = a
                    h_t[1] = b
                    h_t[c_pos] = c
                    s = score(h_t)
                    if s > best_score:
                        best_score = s
                        best_h = h_t.copy()
    
    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
    
    return best_h.tolist()
