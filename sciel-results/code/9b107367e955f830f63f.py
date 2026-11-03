"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from itertools import combinations
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution: (h*h)[k] = sum_i h[i]*h[k-i]
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _try_binary_subset(indices):
    """Try a binary vector with 1s at given indices"""
    h = np.zeros(N)
    for i in indices:
        if 0 <= i < N:
            h[i] = 1.0
    return _compute_ratio(h)

def _greedy_sidon_like(k, deadline):
    """Greedy construction of a set with low autoconvolution peak."""
    best_ratio = 0.0
    best_set = set()
    
    # Start with a few random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        current_set = set()
        # Try to build a set of size k greedily
        for _ in range(k):
            if time.time() > deadline:
                break
            best_idx = -1
            best_max_conv = float('inf')
            
            for idx in range(N):
                if idx in current_set:
                    continue
                test_set = current_set | {idx}
                # Compute max autoconvolution for this set
                h = np.zeros(N)
                for i in test_set:
                    h[i] = 1.0
                conv = np.convolve(h, h, mode='full')
                mc = np.max(conv)
                if mc < best_max_conv:
                    best_max_conv = mc
                    best_idx = idx
            
            if best_idx >= 0:
                current_set.add(best_idx)
        
        if len(current_set) == k:
            ratio = _try_binary_subset(list(current_set))
            if ratio > best_ratio:
                best_ratio = ratio
                best_set = set(current_set)
    
    return best_ratio, list(best_set)

def _local_search(h, deadline, max_iter=5000):
    """Local search by flipping bits and adjusting heights."""
    h = np.array(h, dtype=float)
    best_ratio = _compute_ratio(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try setting h[i] to 0
            old_val = h[i]
            if old_val > 0:
                h[i] = 0.0
                r = _compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    improved = True
                else:
                    h[i] = old_val
            
            # Try setting h[i] to 1 (if currently 0)
            if old_val == 0:
                h[i] = 1.0
                r = _compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    improved = True
                    old_val = 1.0
                else:
                    h[i] = 0.0
        
        if not improved:
            # Try random perturbations
            for _ in range(10):
                if time.time() > deadline:
                    break
                idx = np.random.randint(N)
                h[idx] = np.random.random()
                r = _compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                else:
                    h[idx] = 0.0 if h[idx] > 0.5 else 1.0  # revert to binary-ish
            break
    
    return best_ratio, h

def construct():
    """Construct optimal heights for the second autocorrelation inequality."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_ratio = 0.0
    best_h = np.ones(N)
    best_ratio = _compute_ratio(best_h)
    
    # Phase 1: Search for good binary patterns using Sidon-like heuristic
    for k in [10, 15, 20, 25, 30, 35, 40]:
        if time.time() > deadline - 30:
            break
        ratio, indices = _greedy_sidon_like(k, deadline - 30)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = np.zeros(N)
            for i in indices:
                if 0 <= i < N:
                    best_h[i] = 1.0
    
    # Phase 2: Local search on the best pattern found
    if time.time() < deadline - 20:
        ratio, h = _local_search(best_h, deadline - 20)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h
    
    # Phase 3: Try a few more random binary patterns
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        max_tries = int(remaining_time / 0.01)  # ~100 tries per second
        for _ in range(min(max_tries, 5000)):
            if time.time() > deadline - 5:
                break
            # Random sparse binary pattern
            k = np.random.randint(8, 35)
            indices = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            r = _compute_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h
    
    # Return as list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
