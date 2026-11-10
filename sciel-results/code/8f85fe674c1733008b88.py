"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [1, n]. signs is a numpy array of -1/+1 of length n (0-indexed).
    An AP is (start, step, length) where 0 <= start, start + (length-1)*step < n,
    step >= 1, length >= 1.
    """
    n = len(signs)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For each step d from 1 to n-1, and each residue r in [0, d-1],
    # consider the subsequence signs[r], signs[r+d], signs[r+2d], ...
    # All APs with step d and start congruent to r mod d are contiguous
    # subsequences of this subsequence.
    for d in range(1, n):
        for r in range(d):
            # Indices: r, r+d, r+2d, ... < n
            idx = np.arange(r, n, d)
            if len(idx) <= 1:
                continue
            seq = signs[idx]
            csum = np.cumsum(seq)
            # Max absolute sum over all contiguous subsequences of seq
            # = max(csum) - min(csum) ... no, that's max subarray sum.
            # Actually we need max over all i<=j of |csum[j] - (csum[i-1] if i>0 else 0)|
            # This is equivalent to: max_abs = max( max(csum) - min(csum), ... )
            # Wait, the max absolute sum of a contiguous subarray is:
            # max( max_subarray_sum, -min_subarray_sum )
            # where max_subarray_sum = max(csum) - min(csum_before)
            # Let's just compute it properly.
            
            # For a sequence, the max absolute sum of any contiguous subarray is:
            # max( max(0, max_subarray_sum), max(0, -min_subarray_sum) )
            # But we want the absolute value, so it's max(max_subarray_sum, -min_subarray_sum)
            # if both are non-negative, else just the positive one.
            # Actually: max_abs = max(max_subarray_sum, abs(min_subarray_sum))
            # where max_subarray_sum = max_{i<=j} sum[i..j]
            # and min_subarray_sum = min_{i<=j} sum[i..j]
            
            # max_subarray_sum can be found by: max(csum) - min(csum) where min is taken
            # over elements before the max. More precisely, Kadane's algorithm.
            # But since we already have csum, we can do:
            # max_subarray_sum = max(csum[j] - min(csum[0..j-1])) for j in range(len(csum))
            # with csum[-1] = 0 conceptually.
            
            # Simpler: use the fact that max contiguous subarray sum = max(csum) - min(csum)
            # is NOT correct in general. The correct formula uses running minimum.
            
            # Let's just use a simple O(m) approach with running min/max:
            m = len(seq)
            # max subarray sum
            min_so_far = 0
            max_sum = 0
            for j in range(m):
                if j == 0:
                    cs = csum[0]
                else:
                    cs = csum[j]
                # sum from i to j = csum[j] - csum[i-1] (with csum[-1]=0)
                # We want max over i of (csum[j] - csum[i-1])
                # So for each j, the best i gives csum[j] - min(csum[0..j-1])
                # where csum[-1] = 0
                min_so_far = min(min_so_far, cs if j > 0 else 0)
                # Actually let me redo this properly
                pass
            
            # Let me use a cleaner approach:
            # Build an extended csum with csum_ext[0]=0, csum_ext[k]=csum[k-1] for k>=1
            csum_ext = np.empty(m + 1, dtype=np.int32)
            csum_ext[0] = 0
            csum_ext[1:] = csum
            
            # Max subarray sum = max(csum_ext[j] - min(csum_ext[0..j-1])) for j=1..m
            # Min subarray sum = min(csum_ext[j] - max(csum_ext[0..j-1])) for j=1..m
            
            # Running min and max:
            running_min = np.empty(m + 1, dtype=np.int32)
            running_max = np.empty(m + 1, dtype=np.int32)
            running_min[0] = csum_ext[0]
            running_max[0] = csum_ext[0]
            for j in range(1, m + 1):
                running_min[j] = min(running_min[j-1], csum_ext[j])
                running_max[j] = max(running_max[j-1], csum_ext[j])
            
            # Max subarray sum:
            # For each j from 1 to m: csum_ext[j] - running_min[j-1]
            max_subarray = np.max(csum_ext[1:] - running_min[:-1])
            # Min subarray sum:
            # For each j from 1 to m: csum_ext[j] - running_max[j-1]
            min_subarray = np.min(csum_ext[1:] - running_max[:-1])
            
            max_abs = max(abs(max_subarray), abs(min_subarray))
            if max_abs > max_disc:
                max_disc = max_abs
    
    return max_disc


def _local_search(signs, max_iters=5000):
    """
    Local search: try flipping each position and accept if it reduces max_abs_ap_sum.
    Use a deadline check.
    """
    n = len(signs)
    signs = signs.copy()
    current_max = _max_abs_ap_sum(signs)
    
    start_time = time.time()
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        for i in range(n):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            # Try flipping position i
            signs[i] = -signs[i]
            new_max = _max_abs_ap_sum(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = -signs[i]  # revert
    
    return signs, current_max


def _greedy_construct():
    """
    Greedy construction: assign signs one by one to minimize the current max AP sum.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        best_val = None
        best_sign = 1
        for s in (1, -1):
            signs[i] = s
            # Compute max AP sum for the partial assignment
            # Only APs that are fully assigned matter
            max_disc = 0
            for d in range(1, n):
                for r in range(d):
                    if r >= i or r + d >= i:
                        # This AP might include position i or later
                        # Check if all positions are assigned
                        idx = list(range(r, n, d))
                        if any(j > i for j in idx):
                            continue
                        seq = signs[idx]
                        csum = np.cumsum(seq)
                        csum_ext = np.empty(len(csum) + 1, dtype=np.int32)
                        csum_ext[0] = 0
                        csum_ext[1:] = csum
                        running_min = np.empty(len(csum) + 1, dtype=np.int32)
                        running_max = np.empty(len(csum) + 1, dtype=np.int32)
                        running_min[0] = csum_ext[0]
                        running_max[0] = csum_ext[0]
                        for j in range(1, len(csum) + 1):
                            running_min[j] = min(running_min[j-1], csum_ext[j])
                            running_max[j] = max(running_max[j-1], csum_ext[j])
                        max_subarray = np.max(csum_ext[1:] - running_min[:-1])
                        min_subarray = np.min(csum_ext[1:] - running_max[:-1])
                        max_abs = max(abs(max_subarray), abs(min_subarray))
                        if max_abs > max_disc:
                            max_disc = max_abs
            if best_val is None or max_disc < best_val:
                best_val = max_disc
                best_sign = s
        signs[i] = best_sign
    
    return signs


def construct():
    start_time = time.time()
    
    # Start with a random initialization
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=_N).astype(np.int32)
    
    # Evaluate initial
    current_max = _max_abs_ap_sum(signs)
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search
    signs, current_max = _local_search(signs, max_iters=100)
    if current_max < best_max:
        best_max = current_max
        best_signs = signs.copy()
    
    # Try a few more random restarts
    for seed in [171, 123, 456, 789, 321]:
        if time.time() - start_time > _DEADLINE - 10:
            break
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=_N).astype(np.int32)
        current_max = _max_abs_ap_sum(signs)
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
        
        # Local search from this start
        signs, current_max = _local_search(signs, max_iters=100)
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
    
    # Final check
    if time.time() - start_time > _DEADLINE - 2:
        return best_signs.tolist()
    
    # One more local search pass on the best
    signs, current_max = _local_search(best_signs, max_iters=200)
    if current_max < best_max:
        best_max = current_max
        best_signs = signs.copy()
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
