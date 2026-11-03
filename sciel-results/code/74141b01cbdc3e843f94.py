import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=float)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    ac = np.convolve(h, h, mode='full')
    ac_inf = np.max(ac)
    if ac_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * ac_inf)

def _is_sidon(s):
    """Check if s is a Sidon set (all pairwise sums distinct)."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            s_val = s[i] + s[j]
            if s_val in sums:
                return False
            sums.add(s_val)
    return True

def _find_max_sidon(n, max_size=10):
    """Find a large Sidon set in {0, ..., n-1} using greedy."""
    best = []
    for i in range(n):
        candidate = best + [i]
        if _is_sidon(candidate):
            best = candidate
        if len(best) >= max_size:
            break
    return best

def construct():
    deadline = time.time() + 140
    
    # Find a large Sidon set
    sidon = _find_max_sidon(N, 10)
    k = len(sidon)
    
    # Create heights: equal spikes at Sidon positions
    h = np.zeros(N)
    for pos in sidon:
        h[pos] = 1.0 / k
    
    score = _compute_ratio(h)
    
    # Local optimization
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i, pos in enumerate(sidon):
            for delta in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.03, -0.03]:
                h_new = h.copy()
                h_new[pos] = max(0, h_new[pos] + delta)
                s_new = _compute_ratio(h_new)
                if s_new > score + 1e-10:
                    h = h_new
                    score = s_new
                    improved = True
        if not improved:
            break
    
    return list(h)
# EVOLVE-BLOCK-END
