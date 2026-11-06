import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime parts
    arr = np.arange(lower, max_val + 1, dtype=np.int64)
    smooth_parts = np.ones(nvals, dtype=np.int64)

    for p in _SMALL_PRIMES:
        temp = arr.copy()
        count = np.zeros(nvals, dtype=np.int32)
        
        # Vectorized exponent counting
        while True:
            mask = (temp % p == 0)
            if not np.any(mask):
                break
            temp[mask] //= p
            count[mask] += 1
        
        # Multiply smooth_parts by p^count
        # Using np.power is efficient
        powers = np.power(p, count, dtype=np.int64)
        smooth_parts *= powers

    # Convert to list for fast scalar access
    sp_list = smooth_parts.tolist()

    best_n = lower
    best_count = 999

    # Sliding window
    # Window size is 24
    # We need to track frequency of each distinct factor value
    freq = {}
    distinct_count = 0
    
    # Initialize window for n = lower (indices 1 to 24)
    for i in range(1, 25):
        v = sp_list[i]
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct_count += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = lower
        if best_count == 1:
            return best_n

    # Slide window
    # n goes from lower + 1 to upper
    # offset = n - lower
    # Window for n=lower+offset is indices (offset+1) to (offset+24)
    # Previous window was indices offset to offset+23
    # Remove index offset, add index offset+24
    
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break
            
        val_out = sp_list[offset]
        val_in = sp_list[offset + 24]
        
        # Remove val_out
        cnt_out = freq[val_out]
        if cnt_out == 1:
            del freq[val_out]
            distinct_count -= 1
        else:
            freq[val_out] = cnt_out - 1
            
        # Add val_in
        if val_in in freq:
            freq[val_in] += 1
        else:
            freq[val_in] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
