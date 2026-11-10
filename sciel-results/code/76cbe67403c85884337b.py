"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute sum over all arithmetic progressions in [1, n].
    signs is a numpy array of length n with values -1 or +1.
    An AP is defined by start index s (0-based), step d >= 1, and length L >= 1.
    The sum is signs[s] + signs[s+d] + ... + signs[s+(L-1)*d].
    We consider all APs that fit within [0, n-1].
    """
    n = len(signs)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For each step d, and for each residue r mod d, the elements at positions
    # r, r+d, r+2d, ... form a sequence. Any contiguous subsequence of this
    # sequence is an AP with step d. So we compute max subarray sum (and min)
    # for each such sequence.
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            # Compute cumulative sum
            csum = np.cumsum(seq)
            # Max subarray sum: max(csum[j] - csum[i]) for j > i, where csum[i] is sum of seq[0:i]
            # We can compute this by tracking min prefix so far
            min_prefix = 0
            max_sub = 0
            max_sub_abs = 0
            for val in csum:
                max_sub = max(max_sub, val - min_prefix)
                min_prefix = min(min_prefix, val)
            # Min subarray sum (most negative)
            max_prefix = 0
            min_sub = 0
            for val in csum:
                min_sub = min(min_sub, val - max_prefix)
                max_prefix = max(max_prefix, val)
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_disc:
                max_disc = current_max
    
    return max_disc

def _local_search(signs, max_iters=100000, deadline=85.0):
    """
    Local search: flip individual signs to reduce the maximum absolute AP sum.
    """
    signs = np.array(signs, dtype=np.int32)
    n = len(signs)
    current_max = _max_abs_ap_sum(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try flipping each position
        best_gain = 0
        best_flip = -1
        
        for i in range(n):
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = _max_abs_ap_sum(signs)
            signs[i] = old_sign
            
            if new_max < current_max:
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_flip = i
                    best_new_max = new_max
        
        if best_flip >= 0:
            signs[best_flip] = -signs[best_flip]
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # No single flip improves; try random restarts of local search
            # or break
            break
    
    return best_signs.tolist(), best_max

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with a random initialization
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=_N).astype(np.int32)
    
    current_max = _max_abs_ap_sum(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    # Perform local search
    best_signs, best_max = _local_search(signs, max_iters=50000, deadline=deadline)
    
    # If we have time, try more iterations with different starting points
    while time.time() < deadline - 5:
        # Random restart
        signs = rng.choice([-1, 1], size=_N).astype(np.int32)
        current_max = _max_abs_ap_sum(signs)
        
        # Quick local search
        s, m = _local_search(signs, max_iters=10000, deadline=deadline)
        if m < best_max:
            best_max = m
            best_signs = np.array(s)
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
