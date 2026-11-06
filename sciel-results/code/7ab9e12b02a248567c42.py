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
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        power = p
        while power <= max_val:
            mask = (arr % power == 0)
            if np.any(mask):
                sp_parts[mask] *= p
            power *= p

    # Fast path: check if any window of 24 consecutive values are all equal
    # (distinct count = 1). This is the theoretical minimum.
    # A window starting at index i (1-based in arr, 0-based in sp_parts)
    # covers sp_parts[i] through sp_parts[i+23].
    # All equal iff sp_parts[i] == sp_parts[i+1] == ... == sp_parts[i+23]
    # Equivalent: all 23 adjacent pairs in that range are equal.
    # adj_eq[j] = (sp_parts[j] == sp_parts[j+1]) for j in 0..nvals-2
    # Window starting at index i needs adj_eq[i]..adj_eq[i+22] all True.
    
    adj_eq = (sp_parts[:-1] == sp_parts[1:])  # length nvals-1
    
    # Find runs of True in adj_eq. A run of length >= 23 starting at index i
    # means sp_parts[i..i+23] are all equal, giving window n = lower + i.
    # We need i >= 1 (since window covers n+1..n+24, so sp_parts indices 1..24)
    # and i+23 <= nvals-1, i.e., i <= nvals-24 = upper+24-lower+1-24 = upper-lower+1
    
    # Use cumulative trick: compute for each position how many consecutive True
    # end at that position.
    # Actually, simpler: use a sliding sum on adj_eq (treat True as 1)
    # A window of 23 consecutive Trues starting at adj_eq[i] means sum(adj_eq[i:i+23]) == 23
    
    # Build prefix sum of adj_eq
    adj_eq_int = adj_eq.astype(np.int32)
    prefix = np.zeros(nvals, dtype=np.int32)
    prefix[1:] = np.cumsum(adj_eq_int)
    
    # For window starting at sp_parts index i (i from 1 to nvals-24):
    # Need sum of adj_eq[i..i+22] == 23
    # That's prefix[i+23] - prefix[i] == 23
    # i ranges from 1 to nvals-24
    max_i = nvals - 24
    # Check all i in [1, max_i]
    idx_start = 1
    idx_end = max_i  # inclusive
    if idx_end >= idx_start:
        sums = prefix[idx_start+23:idx_end+23+1] - prefix[idx_start:idx_end+1]
        # Find first i where sums == 23
        matches = np.where(sums == 23)[0]
        if len(matches) > 0:
            first_match = matches[0]
            n_val = lower + idx_start + first_match
            return n_val
    
    # If no window of distinct=1 found, do sliding window for minimum distinct
    sp_list = sp_parts.tolist()
    
    best_n = lower
    best_count = 999
    
    window = {}
    for i in range(1, 25):
        v = sp_list[i]
        window[v] = window.get(v, 0) + 1
    distinct = len(window)
    best_count = distinct
    
    n_steps = upper - lower
    check_interval = 50000
    for offset in range(1, n_steps + 1):
        if offset % check_interval == 0 and time.time() > deadline:
            break
        
        val_out = sp_list[offset]
        c = window[val_out] - 1
        if c == 0:
            del window[val_out]
        else:
            window[val_out] = c
        
        val_in = sp_list[offset + 24]
        window[val_in] = window.get(val_in, 0) + 1
        
        distinct = len(window)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
# EVOLVE-BLOCK-END
