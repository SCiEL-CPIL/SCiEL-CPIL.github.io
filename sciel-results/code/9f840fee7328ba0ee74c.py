import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # sp_parts[i] = product of all prime powers p^k where p < 24 and p^k divides (lower+i)
    
    # We work on the array of actual values
    arr = np.arange(lower, max_val + 1, dtype=np.int64)
    smooth_parts = np.ones(nvals, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Calculate the exponent of p in each number
        temp = arr.copy()
        count = np.zeros(nvals, dtype=np.int32)
        
        # Vectorized exponent counting
        # Loop until no more elements are divisible by p
        # Max exponent for p=2 in 10^6 is ~19. This loop runs ~19 times.
        while True:
            mask = (temp % p == 0)
            if not np.any(mask):
                break
            temp[mask] //= p
            count[mask] += 1
        
        # Multiply smooth_parts by p^count
        # np.power works element-wise
        powers = np.power(p, count)
        smooth_parts *= powers

    # Convert to list for faster access in Python loop? 
    # Or stick with numpy array and index?
    # List access is generally faster in Python than numpy array scalar access.
    sp_list = smooth_parts.tolist()

    best_n = lower
    best_count = 999  # max possible distinct count is small

    # Use a frequency dictionary for the sliding window
    window = defaultdict(int)
    
    # Initial window for n=lower: indices 1 through 24 in sp_list
    # Note: sp_list[0] corresponds to value 'lower'
    # We need n+1 to n+24.
    # If n = lower, we need lower+1 to lower+24.
    # In sp_list, these are indices 1 to 24.
    
    for i in range(1, 25):
        v = sp_list[i]
        window[v] += 1
    distinct = len(window)
    if distinct < best_count:
        best_count = distinct
        best_n = lower
        if best_count == 1:
            return best_n

    # Slide the window
    # n ranges from lower to upper
    # Current n = lower. Window covers sp_list[1..24].
    # Next n = lower + 1. Window covers sp_list[2..25].
    # We remove sp_list[1] and add sp_list[25].
    
    # Total iterations: upper - lower
    # We stop if time is up.
    
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break
        
        # Remove sp_list[offset] (which is the value at n+offset-1, i.e., the one leaving)
        # Wait, let's trace indices carefully.
        # n = lower + offset - 1 was the previous state?
        # Let's define state by `n`.
        # Start: n = lower. Window: sp_list[1] ... sp_list[24].
        # Step 1: n becomes lower + 1.
        # New Window: sp_list[2] ... sp_list[25].
        # Removed: sp_list[1].
        # Added: sp_list[25].
        
        val_out = sp_list[offset]
        window[val_out] -= 1
        if window[val_out] == 0:
            del window[val_out]
        
        val_in = sp_list[offset + 24]
        window[val_in] += 1
        
        distinct = len(window)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
