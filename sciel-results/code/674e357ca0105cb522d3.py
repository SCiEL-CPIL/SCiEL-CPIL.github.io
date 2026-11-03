import numpy as np
import time
import random

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_max_sidon_set(n, max_k=10, trials=1000):
    """Find a Sidon set of maximum size in [0, n-1] using randomized greedy."""
    best_set = None
    best_size = 0
    
    for _ in range(trials):
        current_set = []
        used_sums = set()
        
        # Randomize the order of candidate elements
        candidates = list(range(n))
        random.shuffle(candidates)
        
        for idx in candidates:
            # Check if idx can be added
            valid = True
            for s in current_set:
                if (s + idx) in used_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            if (idx + idx) in used_sums:
                continue
            
            # Tentatively add idx
            current_set.append(idx)
            new_sums = set()
            for s in current_set[:-1]:
                new_sums.add(s + idx)
            new_sums.add(idx + idx)
            
            # Check for conflicts within new_sums? 
            # The check above ensures (s+idx) not in used_sums.
            # But we also need to ensure new_sums don't conflict with each other?
            # No, s1+idx and s2+idx are distinct if s1!=s2.
            # And idx+idx is distinct from s+idx if s!=idx.
            # So we just need to update used_sums.
            
            used_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size == max_k:
                break

    return best_set

def construct():
    # 1. Find the largest Sidon set
    # Try to find size 10 first, then 9, etc.
    # The greedy approach finds *a* set, not necessarily the maximum.
    # But with randomization, it often finds large ones.
    # We can also try to start with a known good set and swap.
    
    sidon_set = _find_max_sidon_set(N, 10, trials=2000)
    
    if sidon_set is None or len(sidon_set) == 0:
        # Fallback to Uniform
        h = np.ones(N) / N
        return list(h)
        
    K = len(sidon_set)
    
    # 2. Use uniform weights, which are optimal for Sidon sets
    h_final = np.zeros(N)
    for idx in sidon_set:
        h_final[idx] = 1.0 / K
        
    return list(h_final)
