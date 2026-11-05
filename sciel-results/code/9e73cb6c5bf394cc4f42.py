import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute E[m] using NumPy
    E = np.ones(MAX_VAL + 1, dtype=np.int32)
    indices = np.arange(MAX_VAL + 1, dtype=np.int32)
    
    for p in small_primes:
        power = p
        while power <= MAX_VAL:
            mask = indices % power == 0
            E[mask] *= p
            power *= p
            if power > MAX_VAL:
                break

    # Convert to Python list for faster access in the sliding window loop
    E_list = E.tolist()

    best_n = N_MIN
    best_count = WINDOW + 1
    
    freq = {}
    current_count = 0
    
    # Initial window
    for i in range(1, WINDOW + 1):
        val = E_list[N_MIN + i]
        count = freq.get(val, 0)
        if count == 0:
            freq[val] = 1
            current_count += 1
        else:
            freq[val] = count + 1
            
    if current_count < best_count:
        best_count = current_count
        best_n = N_MIN
        
    # Slide
    # We break if we find a very low count to save time
    # 5 is a very good score. 4 is likely impossible or extremely rare.
    for n in range(N_MIN + 1, N_MAX + 1):
        # Remove n
        val_out = E_list[n]
        count_out = freq[val_out]
        if count_out == 1:
            del freq[val_out]
            current_count -= 1
        else:
            freq[val_out] = count_out - 1
            
        # Add n + WINDOW
        val_in = E_list[n + WINDOW]
        count_in = freq.get(val_in, 0)
        if count_in == 0:
            freq[val_in] = 1
            current_count += 1
        else:
            freq[val_in] = count_in + 1
            
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count <= 5:
                break
                
        # Deadline check
        if time.time() > DEADLINE:
            break

    return best_n
