"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1  # ~999025

    # Precompute small-prime part for each number in [lower, max_val]
    # Using vectorized power-based approach (inspiration 2/3)
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        power = p
        while power <= max_val:
            mask = (arr % power == 0)
            if np.any(mask):
                sp_parts[mask] *= p
            power *= p

    # Now we need to find n in [lower, upper] minimizing distinct count
    # in sp_parts[n+1-lower .. n+24-lower]
    # Let offset = n - lower, window is sp_parts[offset+1 .. offset+24]
    # So we need distinct counts for windows starting at index 1, 2, ..., upper-lower

    # Convert to a compact integer ID array for faster operations
    # First, get unique values
    unique_vals, inverse = np.unique(sp_parts, return_inverse=True)
    num_unique = len(unique_vals)

    # For the sliding window distinct count, we use a difference-array approach:
    # For each position i in the window range, track when each value first appears
    # and when it last appears. A value contributes to distinct count in window [s, s+23]
    # if its first occurrence in [s, s+23] is at some position and it appears at least once.
    
    # Alternative: Use the fact that distinct count = number of values with freq >= 1
    # We can compute this using a "first occurrence" and "last occurrence" trick.
    
    # For each value v at positions pos_1 < pos_2 < ... < pos_k:
    # v is present in window starting at s iff s <= pos_i <= s+23 for some i
    # i.e., s in [pos_i - 23, pos_i] for some i
    # The union of these intervals gives the range of s where v is present.
    
    # But computing the union for all values is complex. Let's use a different approach:
    # For each window start s, distinct = number of distinct values in sp_parts[s..s+23]
    
    # Since the window is only 24 elements, and we have ~999000 windows,
    # let's try a different vectorized approach:
    
    # Method: For each position i, define first_occ[i] = i if sp_parts[i] != sp_parts[i-1] 
    # (or i=0), else -1. Similarly last_occ.
    # Actually, let's use the standard trick:
    # distinct in window [s, s+w-1] = count of i in [s, s+w-1] such that 
    #   the previous occurrence of sp_parts[i] is before s
    # i.e., prev_occ[i] < s where prev_occ[i] = previous index with same value (or -1)
    
    # Build prev_occ array
    prev_occ = np.full(nvals, -1, dtype=np.int64)
    # For each unique value, find its positions and set prev_occ
    # This is O(n * num_unique) in worst case but num_unique is small
    
    # Actually, let's build it more efficiently:
    # Sort by value, then within each group, set prev_occ
    sorted_idx = np.argsort(inverse, kind='stable')
    sorted_inv = inverse[sorted_idx]
    
    # For each consecutive pair in sorted order with same value, set prev_occ
    # prev_occ[sorted_idx[i]] = sorted_idx[i-1] if sorted_inv[i] == sorted_inv[i-1]
    
    # Vectorized: where sorted_inv[i] == sorted_inv[i-1], set prev_occ[sorted_idx[i]] = sorted_idx[i-1]
    same_as_prev = np.zeros(nvals, dtype=bool)
    same_as_prev[1:] = sorted_inv[1:] == sorted_inv[:-1]
    
    # For those positions, prev_occ[sorted_idx[i]] = sorted_idx[i-1]
    # We need to map back: for i where same_as_prev[i], prev_occ[sorted_idx[i]] = sorted_idx[i-1]
    # This requires scatter, which is tricky. Let's use a loop over unique values.
    
    # Given num_unique is at most a few hundred (products of small prime powers),
    # a loop over unique values is fine.
    
    prev_occ = np.full(nvals, -1, dtype=np.int64)
    for uid in range(num_unique):
        positions = np.where(inverse == uid)[0]
        if len(positions) > 1:
            prev_occ[positions[1:]] = positions[:-1]
    
    # Now for window starting at s (covering indices s..s+23):
    # distinct = number of i in [s, s+23] such that prev_occ[i] < s
    # i.e., prev_occ[i] <= s-1
    
    # For each i, it contributes to windows s where:
    #   s <= i <= s+23  AND  prev_occ[i] < s
    #   => max(prev_occ[i]+1, i-23) <= s <= i
    #   => s in [max(prev_occ[i]+1, i-23), i]
    
    # So we can use a difference array:
    # For each i, add 1 to all s in [max(prev_occ[i]+1, i-23), i]
    # Then the value at position s gives the distinct count for window starting at s.
    
    # But we only care about s in [1, upper-lower] (since n=lower+s, and n+1 to n+24
    # correspond to indices s+1 to s+24, wait let me re-check)
    
    # Window for n: indices (n+1-lower) to (n+24-lower) = (offset+1) to (offset+24)
    # where offset = n - lower
    # So window starts at index offset+1, ends at offset+24
    # s = offset+1, so offset = s-1
    # We need s in [1, upper-lower+1]... wait:
    # offset ranges from 0 to upper-lower
    # s = offset+1 ranges from 1 to upper-lower+1
    # Window is [s, s+23] = [offset+1, offset+24]
    
    # So we need distinct counts for windows starting at s=1, 2, ..., upper-lower+1
    # which is 999001 windows.
    
    # Build difference array for s in [0, nvals] (we'll use indices 1 to upper-lower+1)
    # diff[s] += 1 for s in [lo, hi] means diff[lo] += 1, diff[hi+1] -= 1
    
    n_windows = upper - lower + 1  # number of valid n values
    # s ranges from 1 to n_windows
    # We'll create an array of size n_windows+2 for the difference array
    
    diff = np.zeros(n_windows + 2, dtype=np.int32)
    
    # For each index i in [1, nvals-1] (since windows start at s>=1 and end at s+23 <= nvals-1):
    # Actually i must be such that the window [s, s+23] is valid, i.e., s+23 < nvals
    # s <= nvals-24, and s >= 1
    # i is in [s, s+23], so i is in [1, nvals-1]
    
    # For each i, lo = max(prev_occ[i]+1, i-23), hi = i
    # We need lo >= 1 and hi <= n_windows (since s must be in [1, n_windows])
    # Also s+23 < nvals => s < nvals-23 => s <= nvals-24
    # n_windows = upper-lower+1 = 999001
    # nvals = 999025
    # nvals-24 = 999001 = n_windows, so s can be at most n_windows. Good.
    
    # Vectorized computation:
    # i ranges from 1 to nvals-1 (0-indexed, so indices 1 to nvals-1)
    # But we need i such that the window is valid.
    # i must be at least 1 (s >= 1, i >= s >= 1)
    # i must be at most nvals-1 (s+23 <= nvals-1 => s <= nvals-24, i <= s+23 <= nvals-1)
    
    i_arr = np.arange(1, nvals, dtype=np.int64)  # indices 1 to nvals-1
    prev_vals = prev_occ[i_arr]  # prev_occ for these indices
    i_vals = i_arr
    
    lo = np.maximum(prev_vals + 1, i_vals - 23)
    hi = i_vals
    
    # Clip to valid range [1, n_windows]
    lo = np.maximum(lo, 1)
    hi = np.minimum(hi, n_windows)
    
    # Only include where lo <= hi
    valid = lo <= hi
    
    if np.any(valid):
        lo_valid = lo[valid]
        hi_valid = hi[valid]
        
        # Use np.add.at for scatter addition (handles duplicates)
        # diff[lo] += 1, diff[hi+1] -= 1
        np.add.at(diff, lo_valid, 1)
        np.add.at(diff, hi_valid + 1, -1)
    
    # Prefix sum gives distinct count at each s
    distinct_counts = np.cumsum(diff[:n_windows+1])  # distinct_counts[s] for s=0..n_windows
    # We want distinct_counts[1] to distinct_counts[n_windows]
    
    if time.time() > deadline:
        return lower  # fallback
    
    # Find the minimum
    dc = distinct_counts[1:n_windows+1]  # distinct counts for s=1..n_windows
    min_idx = np.argmin(dc)
    best_s = min_idx + 1  # s value (1-indexed)
    best_n = lower + best_s - 1  # since offset = s-1, n = lower + offset = lower + s - 1
    
    best_count = dc[min_idx]
    
    # Verify: window for n=best_n is indices best_s to best_s+23
    # Let's double-check by computing directly
    window_vals = sp_parts[best_s:best_s+24]
    actual_distinct = len(np.unique(window_vals))
    
    # If there's a mismatch (shouldn't be), fall back to a small search
    if actual_distinct != best_count:
        # Try a few neighbors
        best_n_local = best_n
        best_count_local = actual_distinct
        for dn in range(-10, 11):
            test_n = best_n + dn
            if test_n < lower or test_n > upper:
                continue
            ts = test_n - lower + 1  # s value
            tw = sp_parts[ts:ts+24]
            td = len(np.unique(tw))
            if td < best_count_local:
                best_count_local = td
                best_n_local = test_n
        best_n = best_n_local
    
    return best_n
# EVOLVE-BLOCK-END
