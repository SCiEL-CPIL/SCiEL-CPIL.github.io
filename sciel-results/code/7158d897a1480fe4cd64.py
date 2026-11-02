import numpy as np
import time
import warnings
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0  # seconds

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = np.correlate(h, h, mode='full')
    # The autoconvolution (h*h) at lag k is sum_i h[i]*h[i+k]
    # np.correlate(h, h, 'full') gives cross-correlation, which for autoconv is same as conv
    # Actually np.convolve(h, h, 'full') is the convolution
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup == 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _neg_score(h):
    return -_score(h)

def _sidon_set(n, k, rng):
    """Generate a Sidon set of size k from {0,...,n-1}"""
    best = None
    best_score = -1
    for _ in range(200):
        s = set()
        while len(s) < k:
            candidate = rng.randint(0, n)
            # Check Sidon property: all pairwise sums distinct
            valid = True
            s_copy = list(s) + [candidate]
            sums = set()
            for i in range(len(s_copy)):
                for j in range(i, len(s_copy)):
                    sm = s_copy[i] + s_copy[j]
                    if sm in sums:
                        valid = False
                        break
                    sums.add(sm)
                if not valid:
                    break
            if valid:
                s.add(candidate)
        if len(s) == k:
            # Score: prefer larger spread
            score = max(s) - min(s)
            if score > best_score:
                best_score = score
                best = s
    if best is None:
        # Fallback
        best = set(rng.choice(n, k, replace=False))
    return sorted(best)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _score(best_h)
    
    # Try various Sidon set sizes
    for k in [1, 2, 3, 5, 8, 10, 12, 15]:
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            sidon = _sidon_set(N, k, rng)
            h0 = np.zeros(N)
            h0[sidon] = 1.0
            
            # Try different initial weight patterns
            for weight_pattern in range(3):
                if time.time() > deadline:
                    break
                
                if weight_pattern == 0:
                    h_init = h0.copy()
                elif weight_pattern == 1:
                    h_init = h0 * rng.uniform(0.5, 1.5, k)
                else:
                    h_init = h0 * rng.uniform(0.1, 1.0, k)
                
                result = minimize(
                    _neg_score,
                    h_init,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                h_opt = np.clip(result.x, 0, None)
                val = _score(h_opt)
                
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
    
    # Final check
    if time.time() > deadline:
        return list(best_h)
    
    return list(best_h)
