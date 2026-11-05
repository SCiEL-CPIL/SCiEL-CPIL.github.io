"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1 (position i corresponds to index i+1).
    Returns the max absolute sum.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int8)
    
    max_abs = 0
    
    # For each step d from 1 to n-1
    # For each residue class r in [0, d-1], we have a subsequence
    # We want the max absolute subarray sum of that subsequence
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            # Get the subsequence: signs[r], signs[r+d], signs[r+2d], ...
            # Number of elements: m
            m = (n - 1 - r) // d + 1
            if m <= 0:
                continue
            
            # Extract subsequence
            # Using numpy fancy indexing for speed
            indices = r + d * np.arange(m, dtype=np.int32)
            sub = signs[indices]
            
            # Compute prefix sums
            # cum[0] = 0, cum[i] = sum(sub[0:i])
            # We want max_{0<=i<j<=m} |cum[j] - cum[i]|
            # This is max_{j} (cum[j] - min_{i<j} cum[i]) and min_{j} (cum[j] - max_{i<j} cum[i])
            
            # Efficient way:
            # prefix = np.concatenate(([0], np.cumsum(sub)))
            # But we can do it in one pass or use vectorized ops
            
            prefix = np.empty(m + 1, dtype=np.int32)
            prefix[0] = 0
            np.cumsum(sub, out=prefix[1:])
            
            # Now find max difference prefix[j] - prefix[i] for j > i
            # And min difference
            
            # Max positive subarray sum:
            # min_prefix_so_far[i] = min(prefix[0..i])
            # max_pos = max(prefix[i] - min_prefix_so_far[i-1]) for i=1..m
            
            # We can compute min_prefix_so_far efficiently
            # min_prefix_so_far[i] = min(min_prefix_so_far[i-1], prefix[i])
            
            # Let's use a loop or vectorized approach
            # For m up to 256, a simple loop is fast enough, but let's try vectorized if possible
            
            # Actually, for small m, Python loop is fine. But let's use numpy for the whole thing.
            # min_prefix = np.minimum.accumulate(prefix)
            # But we need min_prefix[i-1] for each i.
            
            min_prefix = np.minimum.accumulate(prefix)
            # min_prefix[i] is min(prefix[0..i])
            # For index i in prefix (1..m), the min of prefix[0..i-1] is min_prefix[i-1]
            
            # max_pos_diff[i] = prefix[i] - min_prefix[i-1] for i=1..m
            # We want max of these
            
            # Let's compute:
            # diffs_pos = prefix[1:] - min_prefix[:-1]
            # max_pos = np.max(diffs_pos)
            
            # Similarly for negative:
            # max_prefix = np.maximum.accumulate(prefix)
            # diffs_neg = prefix[1:] - max_prefix[:-1]
            # min_neg = np.min(diffs_neg)
            
            # current_max = max(abs(max_pos), abs(min_neg))
            
            # To avoid creating too many temporary arrays, let's do it in a tight loop for small m?
            # Or just use the vectorized version. n=256, so m is at most 256.
            # Number of (d, r) pairs is sum_{d=1}^{255} d = 255*256/2 = 32640.
            # Each involves cumsum of length up to 256.
            # This should be fast enough in numpy.
            
            min_pref = np.minimum.accumulate(prefix)
            max_pref = np.maximum.accumulate(prefix)
            
            # For i from 1 to m:
            # pos_diff_i = prefix[i] - min_pref[i-1]
            # neg_diff_i = prefix[i] - max_pref[i-1]
            
            # We want max over i of pos_diff_i and min over i of neg_diff_i
            
            # pos_diffs = prefix[1:] - min_pref[:-1]
            # max_pos = pos_diffs.max()
            
            # neg_diffs = prefix[1:] - max_pref[:-1]
            # min_neg = neg_diffs.min()
            
            # Let's just compute the max absolute directly.
            # Actually, we can just iterate if m is small, but vectorized is cleaner.
            
            pos_diffs = prefix[1:] - min_pref[:-1]
            neg_diffs = prefix[1:] - max_pref[:-1]
            
            max_pos = pos_diffs.max() if m > 0 else 0
            min_neg = neg_diffs.min() if m > 0 else 0
            
            current_max = max(abs(max_pos), abs(min_neg))
            if current_max > max_abs:
                max_abs = current_max
                # Early exit optimization: if we reach n//2, we can't do better than n//2?
                # Actually, the maximum possible discrepancy for n=256 is 256.
                # But we don't know a tight lower bound. Let's not early exit unless max_abs is very large.
    
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    
    # Initialize with Thue-Morse sequence, which has good properties
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs[i] = 1 if t == 0 else -1
    
    # Evaluate initial discrepancy
    best_disc = _max_ap_discrepancy(signs)
    
    if time.time() > deadline:
        return signs.tolist()
    
    # Hill climbing: try to flip individual signs to reduce discrepancy
    # We need to be careful about the cost of _max_ap_discrepancy.
    # Let's estimate the time for one evaluation.
    t0 = time.time()
    _ = _max_ap_discrepancy(signs)
    t1 = time.time()
    eval_time = t1 - t0
    
    # If eval_time is too high, we can't do many iterations.
    # Let's aim for a few hundred iterations if possible.
    
    rng = random.Random(42)
    
    improved = True
    iter_count = 0
    max_iters = 10000  # Will be limited by time
    
    while improved and time.time() < deadline and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try flipping each sign in a random order
        order = np.random.permutation(n)
        
        for i in order:
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
            else:
                signs[i] = old_val
    
    return signs.tolist()
# EVOLVE-BLOCK-END
