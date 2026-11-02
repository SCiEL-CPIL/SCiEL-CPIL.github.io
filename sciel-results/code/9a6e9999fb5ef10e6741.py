import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_ratio(h):
        if np.max(h) == 0:
            return 0.0
        h_norm = h / np.max(h)
        l2_sq = np.sum(h_norm**2)
        l1 = np.sum(h_norm)
        l_inf = np.max(h_norm)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    # Candidate 1: Single Spike
    h_single = np.zeros(N)
    h_single[0] = 1.0
    best_h = h_single.copy()
    best_ratio = compute_ratio(best_h)
    
    # Candidate 2: Uniform
    h_uniform = np.ones(N)
    ratio_uniform = compute_ratio(h_uniform)
    if ratio_uniform > best_ratio:
        best_ratio = ratio_uniform
        best_h = h_uniform.copy()
        
    # Candidate 3: Random Sparse (Sidon-like)
    # Try to find a good sparse set via random search
    for _ in range(500):
        if time.time() > deadline:
            break
        k = np.random.randint(2, 20)
        indices = np.random.choice(N, k, replace=False)
        h_sparse = np.zeros(N)
        h_sparse[indices] = 1.0
        r = compute_ratio(h_sparse)
        if r > best_ratio:
            best_ratio = r
            best_h = h_sparse.copy()
            
    # Local Search on best_h
    step_size = 0.05
    h_current = best_h.copy()
    for _ in range(5000):
        if time.time() > deadline:
            break
        i = np.random.randint(N)
        old_val = h_current[i]
        h_current[i] = np.clip(h_current[i] + np.random.uniform(-step_size, step_size), 0, 1)
        if np.max(h_current) > 0:
            h_current = h_current / np.max(h_current)
        
        ratio = compute_ratio(h_current)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h_current.copy()
        else:
            h_current[i] = old_val
            if np.max(h_current) > 0:
                h_current = h_current / np.max(h_current)
            
    return list(best_h)
