import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    MAX_N = 1000000
    WINDOW = 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_val = 1001
    max_val = MAX_N + WINDOW  # 1000024
    
    # Compute smooth part using a sieve-like approach
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            # All multiples of 'power' get multiplied by p
            smooth[power::power] *= p
            power *= p
    
    # Extract relevant portion
    smooth_arr = smooth[min_val:max_val + 1]  # length 999024
    
    # Use a faster sliding window approach with a plain dict
    # and track distinct count incrementally to avoid len() calls
    
    best_n = 1000
    best_count = WINDOW + 1
    
    # Initial window
    freq = {}
    for j in range(WINDOW):
        val = int(smooth_arr[j])
        freq[val] = freq.get(val, 0) + 1
    
    distinct = len(freq)
    best_count = distinct
    best_n = 1000
    
    # Slide the window - optimized inner loop
    # Convert to list for faster indexing than numpy array
    arr_list = smooth_arr.tolist()
    
    n_windows = 999001  # i from 0 to 999000
    
    for i in range(1, n_windows):
        # Remove element at position i-1
        old_val = arr_list[i - 1]
        c = freq[old_val]
        if c == 1:
            del freq[old_val]
            distinct -= 1
        else:
            freq[old_val] = c - 1
        
        # Add element at position i + WINDOW - 1
        new_val = arr_list[i + WINDOW - 1]
        if new_val in freq:
            freq[new_val] += 1
        else:
            freq[new_val] = 1
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = 1000 + i
        
        # Check time every 65536 iterations
        if (i & 0xFFFF) == 0:
            if time.time() > deadline:
                break
    
    return best_n
