"""Rewritten program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    seq = []
    for i in range(n):
        # Count 1-bits in i
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq.append(1 if c % 2 == 0 else -1)
    return seq

def _max_abs_ap_sum_numpy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [1, n] with any start, positive step, and length >= 1.
    signs is a 1D numpy array of int values (-1 or +1), 0-indexed.
    An AP with start s (1-indexed), step d, length L corresponds to indices
    (s-1), (s-1)+d, (s-1)+2d, ..., (s-1)+(L-1)d, all < n.
    For fixed d and residue r = (s-1) % d, the APs are contiguous subsequences
    of the subsequence signs[r::d].
    """
    n = len(signs)
    max_disc = 1  # singletons give |sum| = 1
    signs = np.asarray(signs, dtype=np.int32)
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                # All contiguous subsequence sums = csum[j] - csum[i-1] for i<=j
                # We need max |csum[j] - csum[i-1]| over all 0<=i<=j<len
                # This equals max(csum) - min(csum) if we consider the "prefix" approach
                # Actually, max subarray sum (absolute) can be found via:
                # max over j of (csum[j] - min_{i<=j} csum[i-1]) where csum[-1]=0
                # Simpler: compute all prefix sums including 0, then max - min of the
                # running differences. Actually the standard approach:
                # Let P[0]=0, P[k]=sum(seq[0..k-1]) for k=1..len(seq)
                # Then any contiguous sum = P[j+1]-P[i] for 0<=i<=j+1
                # Max |P[j+1]-P[i]| = max(P) - min(P) over the full prefix array? No,
                # that's not right either. We need max over i<j of |P[j]-P[i]|.
                # That IS max(P) - min(P) only if the max and min occur in the right order.
                # Actually, max_{i<j} (P[j]-P[i]) = max over j of (P[j] - min_{i<j} P[i])
                # And max_{i<j} (P[i]-P[j]) = max over j of (min_{i<j} P[i]... no)
                # Let's just compute it properly but efficiently.
                # For the absolute value, we need both max subarray sum and min subarray sum.
                # Max subarray sum (Kadane-like) and min subarray sum.
                # But we need the max of |sum|, so max(max_subarray, -min_subarray).
                
                # Efficient approach: prefix sums P[0..m] where m=len(seq)
                # P[0]=0, P[k]=sum(seq[0..k-1])
                # We want max_{0<=i<j<=m} |P[j]-P[i]|
                # = max( max_{i<j}(P[j]-P[i]), max_{i<j}(P[i]-P[j]) )
                # = max( max_j P[j] - min_{i<j} P[i], max_j (-P[j]) + max_{i<j} P[i] )
                # The first term: for each j, P[j] - min_{i<j} P[i]
                # The second term: for each j, -P[j] + max_{i<j} P[i]  (i.e., max_{i<j} P[i] - P[j])
                
                P = np.zeros(len(seq) + 1, dtype=np.int32)
                P[1:] = np.cumsum(seq)
                
                # For each j from 1 to len(seq), compute min_{i<j} P[i] and max_{i<j} P[i]
                # We can do this with a running min and max.
                running_min = np.empty(len(seq) + 1, dtype=np.int32)
                running_max = np.empty(len(seq) + 1, dtype=np.int32)
                running_min[0] = P[0]
                running_max[0] = P[0]
                for j in range(1, len(seq) + 1):
                    running_min[j] = min(running_min[j-1], P[j])
                    running_max[j] = max(running_max[j-1], P[j])
                
                # max_{i<j}(P[j]-P[i]) = max over j of (P[j] - min_{i<j} P[i])
                # But running_min[j] includes P[j] itself, so min_{i<=j} P[i]
                # We need min_{i<j}, which is running_min[j-1] if j>0
                # Let me redo: for j>=1, min_{i<j} P[i] = min(P[0],...,P[j-1])
                
                # Let's compute this more carefully with a vectorized approach
                # prefix_min[j] = min(P[0..j-1]) for j>=1, prefix_min[0] = P[0]
                # prefix_max[j] = max(P[0..j-1]) for j>=1, prefix_max[0] = P[0]
                
                # Using numpy: 
                # cummin of P gives min(P[0..j]) for each j
                # So min_{i<j} P[i] = cummin_P[j-1] for j>=1
                
                cummin_P = np.minimum.accumulate(P)
                cummax_P = np.maximum.accumulate(P)
                
                # For j from 1 to len(seq):
                #   term1[j] = P[j] - cummin_P[j-1]  (max P[j]-P[i] for i<j)
                #   term2[j] = cummax_P[j-1] - P[j]  (max P[i]-P[j] for i<j)
                
                if len(seq) >= 1:
                    term1 = P[1:] - cummin_P[:-1]
                    term2 = cummax_P[:-1] - P[1:]
                    local_max = max(int(np.max(term1)), int(np.max(term2)))
                    if local_max > max_disc:
                        max_disc = local_max
    return max_disc

def _local_search(signs, max_iters=5000, time_limit=80.0):
    """
    Local search: repeatedly find the worst AP and flip a sign in it to reduce max |sum|.
    Uses numpy for speed.
    """
    signs = np.array(signs, dtype=np.int32)
    n = len(signs)
    start_time = time.time()
    current_max = _max_abs_ap_sum_numpy(signs)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if current_max <= 1:
            break
        
        # Find a position to flip that reduces the max
        # Try flipping each position and see if it helps
        # For efficiency, we'll try a random subset of positions
        best_pos = -1
        best_new_max = current_max
        
        # Sample some positions to try
        rng = np.random.default_rng()
        candidates = rng.choice(n, size=min(50, n), replace=False)
        
        for pos in candidates:
            signs[pos] = -signs[pos]
            new_max = _max_abs_ap_sum_numpy(signs)
            signs[pos] = -signs[pos]  # revert
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_pos = pos
        
        if best_pos >= 0:
            signs[best_pos] = -signs[best_pos]
            current_max = best_new_max
        else:
            # No single flip helps; try a random perturbation to escape local minimum
            pos = rng.integers(n)
            signs[pos] = -signs[pos]
            current_max = _max_abs_ap_sum_numpy(signs)
    
    return signs, current_max

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with Thue-Morse
    signs = _thue_morse(_N)
    signs_arr = np.array(signs, dtype=np.int32)
    
    best_signs = signs_arr.copy()
    best_max = _max_abs_ap_sum_numpy(best_signs)
    
    # Run local search with multiple restarts
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Local search from current state
        signs_arr, cur_max = _local_search(
            best_signs, 
            max_iters=2000, 
            time_limit=deadline - time.time() - 5.0
        )
        
        if cur_max < best_max:
            best_max = cur_max
            best_signs = signs_arr.copy()
        
        # If we've hit a good value, try to polish with more iterations
        if best_max <= 3:
            signs_arr, cur_max = _local_search(
                best_signs,
                max_iters=5000,
                time_limit=deadline - time.time() - 5.0
            )
            if cur_max < best_max:
                best_max = cur_max
                best_signs = signs_arr.copy()
        
        # Perturb for restart: flip a few random signs
        n_flips = rng.integers(1, 10)
        flip_positions = rng.choice(_N, size=n_flips, replace=False)
        for p in flip_positions:
            best_signs[p] = -best_signs[p]
        
        # Check deadline
        if time.time() > deadline - 10.0:
            # Do one final local search with remaining time
            signs_arr, cur_max = _local_search(
                best_signs,
                max_iters=3000,
                time_limit=deadline - time.time() - 2.0
            )
            if cur_max < best_max:
                best_max = cur_max
                best_signs = signs_arr.copy()
            break
    
    # Final polish if time remains
    if time.time() < deadline - 2.0:
        signs_arr, cur_max = _local_search(
            best_signs,
            max_iters=5000,
            time_limit=deadline - time.time() - 1.0
        )
        if cur_max < best_max:
            best_max = cur_max
            best_signs = signs_arr.copy()
    
    # Return as list of Python integers
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
