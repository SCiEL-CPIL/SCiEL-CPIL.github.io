"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Use more of the 90s budget

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size
    
    # Step 1: Precompute the "small-prime part" using NumPy for speed
    nvals = max_val - lower + 1
    sp_part = np.ones(nvals, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        # For each power p^k, multiply sp_part by p for all numbers divisible by p^k
        power = p
        while power <= max_val:
            first_mult_power = ((lower + power - 1) // power) * power
            if first_mult_power > max_val:
                break
            idx_array_power = np.arange(first_mult_power, max_val + 1, power) - lower
            sp_part[idx_array_power] *= p
            power *= p
    
    if time.time() > deadline:
        return lower

    # Step 2: Use a more efficient approach
    # Convert to int32 to save memory and improve cache locality
    sp_arr = sp_part.astype(np.int64)
    
    # Get unique values and their indices for potential optimization
    # But first, let's try a vectorized approach for small windows
    
    # Strategy: For each position, we want to count distinct values in [i+1, i+24]
    # We can use a sliding window with NumPy operations
    
    # Convert to a list for the sliding window (faster than numpy scalar access)
    sp_list = sp_arr.tolist()
    
    best_n = lower
    best_count = float('inf')
    
    # Use a dict for frequency counting - optimize with local variables
    freq = {}
    current_distinct = 0
    
    # Initialize window for n=lower (indices 1 to 24)
    for i in range(1, window_size + 1):
        val = sp_list[i]
        cnt = freq.get(val, 0)
        if cnt == 0:
            current_distinct += 1
        freq[val] = cnt + 1
        
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower
        
    # Slide window with optimized local variable access
    sp_l = sp_list
    freq_get = freq.get
    ws = window_size
    
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        out_idx = n - lower
        in_idx = n + ws - lower
        
        val_out = sp_l[out_idx]
        cnt_out = freq[val_out]
        if cnt_out == 1:
            del freq[val_out]
            current_distinct -= 1
        else:
            freq[val_out] = cnt_out - 1
            
        val_in = sp_l[in_idx]
        cnt_in = freq_get(val_in, 0)
        if cnt_in == 0:
            current_distinct += 1
        freq[val_in] = cnt_in + 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            # If we found a very good solution, we can potentially stop early
            # The theoretical minimum is 1 (all 24 numbers have the same small-prime part)
            # But that's extremely unlikely. Let's set a practical threshold.
            if best_count <= 3:
                # Very good, but keep searching for a bit more
                pass

    return best_n
# EVOLVE-BLOCK-END
