"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _max_overlap_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    return np.max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap_fast(h)

def _fix_sum(h, N):
    """Ensure h sums to exactly N/2 and is in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            order = np.argsort(1.0 - h)
            for idx in order:
                if diff <= 1e-14:
                    break
                add = min(diff, 1.0 - h[idx])
                h[idx] += add
                diff -= add
        else:
            order = np.argsort(-h)
            for idx in order:
                if -diff <= 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    return h

def _try_alternating(N):
    """Try alternating high/low values with optimal a=0.5."""
    a = 0.5
    h = np.zeros(N)
    h[0::2] = a
    h[1::2] = 1 - a
    return _fix_sum(h, N)

def _local_search_fast(h, max_iter=5000, step_size=0.02, time_limit=15.0):
    """Fast local search using FFT-based scoring."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-step_size, step_size)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
    
    return current, current_score

def _refine_with_scipy(h, time_budget=20.0):
    """Refine h using scipy minimize with penalty for constraints."""
    start = time.time()
    N = len(h)
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = np.clip(x, 0, 1)
        sum_penalty = (np.sum(h) - N/2.0) ** 2 * 1000
        return _score(h) + sum_penalty
    
    # Try multiple starting points
    best_h = h.copy()
    best_score = _score(h)
    
    for trial in range(3):
        if time.time() - start > time_budget - 2:
            break
        if trial == 0:
            x0 = h.copy()
        elif trial == 1:
            x0 = h + np.random.normal(0, 0.05, N)
            x0 = np.clip(x0, 0, 1)
        else:
            x0 = h + np.random.normal(0, 0.1, N)
            x0 = np.clip(x0, 0, 1)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            h_new = np.clip(res.x, 0, 1)
            h_new = _fix_sum(h_new, N)
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new
        except Exception:
            pass
    
    return best_h, best_score

def _final_polish(h, time_budget=10.0):
    """High-precision polish with very small steps."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    # Use very small steps for precision
    for it in range(3000):
        if time.time() - start > time_budget:
            break
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-0.005, 0.005)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-13:
            current = new_h
            current_score = new_score
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin from 150s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with alternating pattern (known good: N/4 = 25.0)
    best_h = _try_alternating(N)
    best_score = _score(best_h)
    
    # Local search from best
    if time.time() < deadline - 50:
        h1, s1 = _local_search_fast(best_h, max_iter=5000, step_size=0.02, time_limit=20.0)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    
    # Refine with scipy (multiple starts)
    if time.time() < deadline - 25:
        h_ref, s_ref = _refine_with_scipy(best_h, time_budget=20.0)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref
    
    # More local search after scipy refinement
    if time.time() < deadline - 15:
        h2, s2 = _local_search_fast(best_h, max_iter=3000, step_size=0.01, time_limit=10.0)
        if s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        h3, s3 = _final_polish(best_h, time_budget=8.0)
        if s3 < best_score:
            best_score = s3
            best_h = h3
    
    # Ensure exact sum and bounds
    best_h = _fix_sum(best_h, N)
    
    return best_h
# EVOLVE-BLOCK-END
