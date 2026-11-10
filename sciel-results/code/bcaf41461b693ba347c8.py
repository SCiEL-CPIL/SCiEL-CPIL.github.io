# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 87.0  # safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute sum over all arithmetic progressions in signs.
    signs: numpy array of int32, length 256.
    Uses the observation that for step d, positions congruent to r mod d form
    a contiguous subsequence, and APs within that are contiguous subarrays.
    """
    n = len(signs)
    max_disc = 1  # singletons always give |sum| = 1
    
    for d in range(2, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) < 2:
                continue
            csum = np.cumsum(seq)
            # max subarray sum and min subarray sum
            # Using Kadane-like with prefix: max(csum) - min(csum before)
            # Actually for max absolute, we need max(csum - min_prefix) and min(csum - max_prefix)
            # Simpler: compute max subarray sum and min subarray sum
            # max subarray sum = max over i of (csum[i] - min(csum[0..i-1]))
            # But we also need the min (most negative) subarray sum.
            # Let's just compute both.
            
            # For max subarray sum:
            min_prefix = np.minimum.accumulate(csum)
            # But we need min of csum[0..i-1] for each i, which is a bit different
            # Actually, max subarray sum = max(csum) - min(csum) is NOT correct in general
            # We need the proper Kadane or prefix-based approach.
            
            # Proper approach:
            # max_subarray = max_{i>=j} (csum[i] - csum[j-1]) where csum[-1]=0
            # Let's use the standard prefix approach:
            # Define extended_csum = [0, csum[0], csum[1], ...]
            # max_subarray = max(extended_csum) - min(extended_csum) is still not right
            # because we need the min to come before the max.
            
            # Let's just use a simple loop for correctness, but vectorize where possible.
            # For speed, use the fact that n is at most 256/d which is small for large d.
            
            # Actually, for small seq lengths, a simple approach works.
            # Let me use the standard O(n) Kadane for max and min subarray.
            
            # Max subarray sum
            max_ending_here = 0
            max_so_far = 0
            for x in seq:
                max_ending_here = max(0, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            
            # Min subarray sum
            min_ending_here = 0
            min_so_far = 0
            for x in seq:
                min_ending_here = min(0, min_ending_here + x)
                min_so_far = min(min_so_far, min_ending_here)
            
            cur_max = max(max_so_far, -min_so_far)
            if cur_max > max_disc:
                max_disc = cur_max
                if max_disc >= 10:  # early exit if already bad
                    return max_disc
    
    return max_disc

def _greedy_construct():
    """
    Greedy construction: build the sequence one element at a time.
    At each position, choose the sign that minimizes the maximum discrepancy
    over all APs ending at that position.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int32)
    
    for pos in range(n):
        best_sign = 1
        best_score = float('inf')
        
        for s in (1, -1):
            signs[pos] = s
            # Evaluate the max discrepancy for APs that include position pos
            # and have pos as the last element (or any AP containing pos)
            # For efficiency, only check APs where pos is the last element
            # i.e., APs: pos - (k-1)*d, pos - (k-2)*d, ..., pos for various d and k
            
            local_max = 1
            for d in range(1, pos + 1):
                # AP ending at pos with step d: pos, pos-d, pos-2d, ...
                # The AP is: pos - (k-1)*d, ..., pos-d, pos
                # Start: pos - (k-1)*d >= 0, so k-1 <= pos/d
                start = pos
                while start >= 0:
                    # Compute sum of signs[start::d] from start to pos
                    # Actually, the AP is start, start+d, start+2d, ..., pos
                    # where pos - start is divisible by d
                    s_sum = 0
                    for i in range(start, pos + 1, d):
                        s_sum += signs[i]
                    abs_s = abs(s_sum)
                    if abs_s > local_max:
                        local_max = abs_s
                    start -= d
            
            if local_max < best_score:
                best_score = local_max
                best_sign = s
        
        signs[pos] = best_sign
    
    return signs

def _local_search(signs, time_limit):
    """
    Local search: flip individual signs to reduce max discrepancy.
    Uses simulated annealing style with decreasing temperature.
    """
    n = _N
    current_signs = signs.copy()
    current_max = _max_abs_ap_sum(current_signs)
    best_signs = current_signs.copy()
    best_max = current_max
    
    start_time = time.time()
    temp = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if temp < 0.01:
            break
        
        # Pick a random position to flip
        pos = np.random.randint(n)
        old_sign = current_signs[pos]
        new_sign = -old_sign
        current_signs[pos] = new_sign
        
        new_max = _max_abs_ap_sum(current_signs)
        
        delta = new_max - current_max
        
        if delta <= 0 or np.random.random() < np.exp(-delta / temp):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = current_signs.copy()
        else:
            current_signs[pos] = old_sign
        
        temp *= cooling
    
    return best_signs, best_max

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    signs = _greedy_construct()
    current_max = _max_abs_ap_sum(signs)
    
    # Phase 2: Local search with time budget
    remaining = _DEADLINE - (time.time() - start_time) - 2.0  # 2s safety
    if remaining > 1.0:
        signs, current_max = _local_search(signs, remaining)
    
    # Convert to list of Python ints
    return [int(s) for s in signs]
# EVOLVE-BLOCK-END
