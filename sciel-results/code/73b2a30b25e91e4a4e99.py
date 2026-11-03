import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
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

def _find_sidon_set(n_elements, max_index):
    """
    Find a set of 'n_elements' indices in [0, max_index] that form a Sidon set
    (all pairwise sums are distinct).
    """
    current_set = []
    used_sums = set()
    
    for i in range(max_index + 1):
        if len(current_set) == n_elements:
            break
        # Check if adding i creates a conflict
        conflict = False
        for j in current_set:
            s = i + j
            if s in used_sums:
                conflict = True
                break
        if not conflict:
            # Add i to set
            current_set.append(i)
            # Update used sums
            for j in current_set:
                used_sums.add(i + j)
                
    return current_set

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Strategy: Use a Sidon set of indices to maximize the ratio.
    # Score for a Sidon set of size K is K / Delta.
    # Delta = 0.01.
    # We want to maximize K.
    # Max K for N=50 is around 7.
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try different sizes K, starting from high to low
    # We can try up to 10, but Sidon sets might not exist for large K.
    for K in range(10, 0, -1):
        sidon_indices = _find_sidon_set(K, N - 1)
        if len(sidon_indices) < K:
            continue
            
        h = np.zeros(N)
        for idx in sidon_indices:
            h[idx] = 1.0 / K
            
        score = _score_raw(h)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # If we found a valid Sidon set, it's likely optimal for this K.
        # We can break if we found a large K, but let's check if a larger K is possible.
        # Since we iterate downwards, the first valid one is the largest K.
        # However, a non-Sidon set with larger K might still be good?
        # No, Sidon is the theoretical max for non-overlapping peaks.
        # If we can't find a Sidon set of size K, we might still get a good score with overlapping peaks,
        # but it will be less than K/Delta.
        # Given the jump from 100 (K=1) to 700 (K=7), finding the largest Sidon set is key.
        
        if len(sidon_indices) == K:
            # Found a Sidon set of size K.
            # This is likely the best we can do for this K.
            # We can return immediately as larger K are tried first.
            return list(best_h)
            
    # Fallback if no Sidon set found (should not happen for K=1)
    if best_score == 0.0:
        best_h[0] = 1.0
        best_score = _score_raw(best_h)
        
    return list(best_h)
