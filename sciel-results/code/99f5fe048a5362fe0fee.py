"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50
DELTA = 1.0 / (2 * N)

def _is_golomb(ruler):
    """Check if the set of indices forms a Golomb ruler (unique differences)."""
    diffs = set()
    for i in range(len(ruler)):
        for j in range(i):
            d = ruler[i] - ruler[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _find_golomb_ruler(n, target_size):
    """Find a Golomb ruler of target_size in range(n)."""
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        new_ruler = ruler + [i]
        if _is_golomb(new_ruler):
            ruler = new_ruler
            if len(ruler) >= target_size:
                break
    return ruler

def _optimize_weights(ruler, iterations=1000, lr=0.01):
    """Optimize weights for a given Golomb ruler to maximize the ratio."""
    M = len(ruler)
    # Initialize weights
    w = np.ones(M)
    
    # Precompute pair indices for convolution
    # c_k = sum_{i,j} w_i w_j delta(i-j-k)
    # For Golomb ruler, each k != 0 has at most 1 pair.
    # c_0 = sum w_i^2
    
    best_w = w.copy()
    best_ratio = 0.0
    
    for _ in range(iterations):
        # Calculate c_k
        c = np.zeros(2*N - 1)
        for i in range(M):
            for j in range(M):
                k = ruler[i] - ruler[j] + (N - 1) # shift to index 0..2N-2
                c[k] += w[i] * w[j]
        
        l2_sq = np.sum(w**2)
        l1 = np.sum(w)
        max_c = np.max(c)
        
        if l1 == 0 or max_c == 0:
            continue
            
        ratio = l2_sq / (l1 * max_c)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_w = w.copy()
            
        # Gradient step
        # We want to maximize R = l2_sq / (l1 * max_c)
        # Simplify: fix l1 = 1 (normalize w)
        # Then maximize l2_sq / max_c
        # grad l2_sq = 2w
        # grad max_c is tricky. Approximate by grad of the max term.
        
        # Find index of max c
        k_max = np.argmax(c)
        
        # Calculate gradient of max_c w.r.t w
        grad_max_c = np.zeros(M)
        # c[k_max] = sum_{i,j: r_i-r_j=k_max-(N-1)} w_i w_j
        # For Golomb, this is a single pair (i*, j*) or i*=j* if k_max=0
        # Actually, we just need the contribution.
        # Simpler: use finite difference or just perturb.
        
        # Normalization constraint: sum(w) = 1
        # Project w to sum=1
        w = w / np.sum(w)
        
        # Simple hill climbing: perturb one weight
        i_rand = random.randint(0, M - 1)
        delta = random.uniform(-0.1, 0.1)
        w_new = w.copy()
        w_new[i_rand] += delta
        if w_new[i_rand] < 0:
            w_new[i_rand] = 0
        w_new = w_new / np.sum(w_new)
        
        # Evaluate new
        c_new = np.zeros(2*N - 1)
        for i in range(M):
            for j in range(M):
                k = ruler[i] - ruler[j] + (N - 1)
                c_new[k] += w_new[i] * w_new[j]
        l2_new = np.sum(w_new**2)
        max_c_new = np.max(c_new)
        ratio_new = l2_new / max_c_new if max_c_new > 0 else 0
        
        if ratio_new > best_ratio:
            w = w_new
            best_ratio = ratio_new
            best_w = w_new.copy()
            
    return best_w, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_h = None
    best_ratio = 0.0
    
    # Try different Golomb ruler sizes
    # Larger M allows for better distribution?
    # But M is limited by N.
    for ts in [10, 9, 8, 7, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        ruler = _find_golomb_ruler(N, ts)
        if len(ruler) < 2:
            continue
            
        w_opt, ratio_opt = _optimize_weights(ruler, iterations=200)
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            h = np.zeros(N)
            for idx, val in zip(ruler, w_opt):
                h[idx] = val
            best_h = h.copy()
            
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
