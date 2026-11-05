"""Program for low discrepancy on all arithmetic progressions.
Uses NumPy-vectorized evaluation, hill-climbing with simulated annealing,
and a final polish pass.
"""
import numpy as np
import time

def _max_ap_discrepancy(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions.
    
    signs: numpy array of shape (n,) with values -1 or +1
    Returns: the maximum absolute sum (integer)
    """
    n = len(signs)
    max_disc = 0
    
    # For each step size, compute all AP sums
    # An AP is defined by (start, step, length) where start in [0, n-1], step >= 1,
    # and start + (length-1)*step < n
    # The sum is sum_{k=0}^{length-1} signs[start + k*step]
    
    # For step=1: these are contiguous subarrays
    # For step>1: these are strided subarrays
    
    # We can compute all AP sums efficiently using cumulative sums for each step.
    # For a fixed step s, the indices 0, s, 2s, ... form one "column",
    # indices 1, 1+s, 1+2s, ... form another, etc.
    # For each such column, we can compute all subarray sums via prefix sums.
    
    for step in range(1, n):
        # For each residue class mod step
        for start in range(step):
            # Get all indices: start, start+step, start+2*step, ...
            idx = np.arange(start, n, step)
            if len(idx) <= 1:
                continue
            vals = signs[idx]
            # Compute all subarray sums
            prefix = np.concatenate(([0], np.cumsum(vals)))
            # Sum from i to j-1 is prefix[j] - prefix[i]
            # We want max over all i < j of |prefix[j] - prefix[i]|
            # This is max(prefix) - min(prefix) for the range, but we need
            # max over pairs where i < j, which is just max(prefix) - min(prefix)
            # since we can always pick the max after the min or vice versa.
            # Actually, we need max_{i<j} (prefix[j] - prefix[i]) and
            # max_{i<j} (prefix[i] - prefix[j]), i.e., max_disc = max(max_prefix_diff, -min_prefix_diff)
            # But simpler: the max absolute subarray sum = max(prefix) - min(prefix)
            # Wait, that's not quite right. We need max over i<j of |prefix[j]-prefix[i]|.
            # This equals max(max_{i<j}(prefix[j]-prefix[i]), max_{i<j}(prefix[i]-prefix[j]))
            # = max(max_prefix - min_prefix_before_max, ...) 
            # Actually, the maximum absolute difference between any two elements where
            # the larger index has the larger value, or the smaller index has the larger value.
            # Simpler: just compute max(prefix) - min(prefix). This is an upper bound
            # and actually equals the answer because we can always find i<j with
            # prefix[j] >= prefix[i] achieving max(prefix)-min(prefix) if the max comes
            # after the min, or we need to be more careful.
            # Let's just use: max subarray sum (can be negative) -> we want max abs.
            # max abs subarray sum = max(max_subarray_sum, -min_subarray_sum)
            # where max_subarray_sum = max_j (prefix[j] - min_{i<j} prefix[i])
            # and min_subarray_sum = min_j (prefix[j] - max_{i<j} prefix[i])
            
            # Efficient: 
            # max_subarray_sum:
            min_prefix = np.minimum.accumulate(prefix)
            max_diff = np.max(prefix - min_prefix)
            
            # min_subarray_sum:
            max_prefix = np.maximum.accumulate(prefix)
            min_diff = np.min(prefix - max_prefix)
            
            disc = max(max_diff, -min_diff)
            if disc > max_disc:
                max_disc = disc
    
    return int(max_disc)

def _max_ap_discrepancy_fast(signs, current_max):
    """Fast check: return the max AP discrepancy, but exit early if it exceeds current_max+1."""
    n = len(signs)
    max_disc = 0
    
    for step in range(1, n):
        for start in range(step):
            idx = np.arange(start, n, step)
            if len(idx) <= 1:
                continue
            vals = signs[idx]
            prefix = np.concatenate(([0], np.cumsum(vals)))
            
            min_prefix = np.minimum.accumulate(prefix)
            max_diff = np.max(prefix - min_prefix)
            
            max_prefix = np.maximum.accumulate(prefix)
            min_diff = np.min(prefix - max_prefix)
            
            disc = max(max_diff, -min_diff)
            if disc > max_disc:
                max_disc = disc
            if max_disc > current_max + 1:
                return max_disc
    
    return int(max_disc)

def construct():
    n = 256
    deadline = time.time() + 55.0  # Safety margin
    
    # Start with a good initial pattern
    # Thue-Morse sequence has low discrepancy properties
    # Generate Thue-Morse: t(i) = parity of number of 1s in binary representation of i
    def thue_morse(i):
        return bin(i).count('1') % 2
    
    signs = np.array([1 if thue_morse(i) == 0 else -1 for i in range(n)], dtype=np.int8)
    
    current_disc = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Precompute all (step, start) pairs for iteration
    ap_pairs = []
    for step in range(1, n):
        for start in range(step):
            ap_pairs.append((step, start))
    
    # Hill climbing with simulated annealing
    import random
    rng = random.Random(42)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    # Count iterations to estimate speed
    iter_count = 0
    
    while time.time() < deadline - 5.0:
        # Try a random flip
        pos = rng.randrange(n)
        signs[pos] = -signs[pos]
        new_disc = _max_ap_discrepancy_fast(signs, best_disc)
        
        if new_disc <= best_disc:
            # Accept
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
            T = max(T_min, T * cooling)
        else:
            # Maybe accept with probability exp(-delta/T)
            delta = new_disc - best_disc
            if delta < T and rng.random() < np.exp(-delta / T):
                T = max(T_min, T * cooling)
            else:
                # Reject
                signs[pos] = -signs[pos]
        
        iter_count += 1
        
        # Occasional restart if stuck
        if iter_count % 5000 == 0 and best_disc > 4:
            # Try a few random restarts
            for _ in range(3):
                random_signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
                rd = _max_ap_discrepancy(random_signs)
                if rd < best_disc:
                    best_disc = rd
                    best_signs = random_signs.copy()
                    signs = random_signs.copy()
                    T = 5.0
                else:
                    signs[pos] = -signs[pos]  # restore
                    break
    
    # Final polish: strict local search
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for pos in range(n):
            signs = best_signs.copy()
            signs[pos] = -signs[pos]
            new_disc = _max_ap_discrepancy_fast(signs, best_disc)
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs
                improved = True
    
    return list(best_signs.astype(int))
