"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math

_N = 256
_DEADLINE = 55.0  # safety margin vs 90s kill

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    n = _N
    
    # 1. Count total APs
    M = 0
    for d in range(1, n + 1):
        for r in range(d):
            L = (n - 1 - r) // d + 1
            if L > 0:
                M += L * (L + 1) // 2
                
    # 2. Initialize structures
    index_to_aps_lists = [[] for _ in range(n)]
    ap_sums = np.zeros(M, dtype=np.float64)
    
    # Initial signs: random or alternating
    # Alternating often gives lower discrepancy for small n
    s = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n)], dtype=np.float64)
    
    # 3. Fill index_to_aps and ap_sums
    k = 0
    for d in range(1, n + 1):
        for r in range(d):
            seq = list(range(r, n, d))
            L = len(seq)
            if L == 0:
                continue
            
            for i in range(L):
                for j in range(i, L):
                    # AP is seq[i : j+1]
                    # Update index_to_aps
                    # And sum
                    sm = 0.0
                    for idx in seq[i:j+1]:
                        index_to_aps_lists[idx].append(k)
                        sm += s[idx]
                    ap_sums[k] = sm
                    k += 1
                    
    # Convert index_to_aps to numpy arrays for fast fancy indexing
    index_to_aps = [np.array(lst) for lst in index_to_aps_lists]
    
    # Initial max
    cur_w = float(np.max(np.abs(ap_sums)))
    best_w = cur_w
    best_s = s.copy()
    
    # 4. Simulated Annealing
    T = 1.0
    # Pre-compute abs_sums buffer to avoid allocation
    abs_sums = np.abs(ap_sums)
    
    while time.time() < deadline - 1.0:
        idx = rng.randrange(n)
        old_val = s[idx]
        
        # Flip
        s[idx] = -old_val
        
        # Update sums
        # Change is -2 * old_val
        delta = -2.0 * old_val
        if len(index_to_aps[idx]) > 0:
            ap_sums[index_to_aps[idx]] += delta
            
        # Compute new max
        np.abs(ap_sums, out=abs_sums)
        new_w = float(np.max(abs_sums))
        
        # Acceptance
        if new_w <= cur_w:
            cur_w = new_w
            if new_w < best_w:
                best_w = new_w
                best_s = s.copy()
        else:
            prob = math.exp(-(new_w - cur_w) / T)
            if rng.random() < prob:
                cur_w = new_w
            else:
                # Revert
                s[idx] = old_val
                ap_sums[index_to_aps[idx]] -= delta
        
        # Cool down
        T = max(0.01, T * 0.999)
        
    return [int(x) for x in best_s]
# EVOLVE-BLOCK-END
