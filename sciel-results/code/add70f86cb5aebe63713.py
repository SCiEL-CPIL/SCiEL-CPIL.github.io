import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
TIME_LIMIT = 140.0

def _autoconv_offdiag(h):
    # Returns the max absolute value of autocorrelation excluding lag 0
    conv = np.convolve(h, h, mode='full')
    # Lag 0 is at index N-1 in 'full' mode
    # We want max of conv[0:N-1] and conv[N:2N-1]
    if N == 1:
        return 0.0
    return np.max(np.abs(np.concatenate([conv[:N-1], conv[N:]])))

def _score(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 0.0
    l2_sq = np.sum(h**2)
    psl = _autoconv_offdiag(h)
    if psl < 1e-10:
        # If PSL is 0, it's a delta function or similar, ratio is infinite?
        # But for N>1, PSL is rarely 0 unless support is 1.
        return l2_sq / l1 # effectively infinite if we consider denominator 0
    return l2_sq / (l1 * psl)

def _cost(h):
    # Minimize PSL / L2^2 ? Or just PSL?
    # We want to maximize L2^2 / (L1 * PSL).
    # Fix L1=1. Maximize L2^2 / PSL.
    # Minimize PSL / L2^2.
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 < 1e-10: return np.inf
    hn = h / l1
    l2_sq = np.sum(hn**2)
    psl = _autoconv_offdiag(hn)
    if l2_sq < 1e-10: return np.inf
    return psl / l2_sq

def _greedy_sidon(target_size, max_val):
    S = [0]
    diffs = set()
    for _ in range(target_size - 1):
        best_cand = -1
        best_max_diff_mult = float('inf')
        # We just want a valid Sidon set, so mult must be 0 for new diffs
        # Greedy: pick smallest cand that doesn't create duplicate diffs
        for cand in range(1, max_val + 1):
            if cand in S: continue
            new_diffs = []
            valid = True
            for s in S:
                d = abs(cand - s)
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                # Check if new_diffs have duplicates among themselves?
                # Sidon set definition: all |a-b| are distinct.
                # So new_diffs must be distinct from each other and from old diffs.
                if len(set(new_diffs)) == len(new_diffs):
                    best_cand = cand
                    diffs.update(new_diffs)
                    S.append(cand)
                    break
        if best_cand == -1:
            break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Try to find a good Sidon set
    # Max Sidon size for N=50 is approx 7-8
    for target_size in [7, 8, 6, 5]:
        if time.time() > deadline - 20: break
        S = _greedy_sidon(target_size, N-1)
        if len(S) < target_size:
            continue
            
        # Optimize weights for this support
        k = len(S)
        bounds = [(0.0, 1.0)] * k
        
        def obj(weights):
            h = np.zeros(N)
            h[S] = weights
            return _cost(h)
            
        # Use DE to find good weights
        try:
            res = differential_evolution(
                obj, bounds, 
                maxiter=200, 
                seed=42,
                polish=True,
                tol=1e-8
            )
            h_opt = np.zeros(N)
            h_opt[S] = res.x
            h_opt = h_opt / h_opt.sum()
            sc = _score(h_opt)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Also try local refinement of the best found so far
    # ... (optional, if time permits)
    
    return [float(x) for x in best_h]
