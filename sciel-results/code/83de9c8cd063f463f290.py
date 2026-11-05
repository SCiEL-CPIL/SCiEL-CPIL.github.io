import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m]
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
        
    # Use a list for faster indexing if values were small, but they can be large.
    # Use a dictionary for frequency counts.
    # To make it even faster, we can map the extracted values to smaller IDs?
    # The values are products of primes < 24.
    # Max value is $23 \times 19 \times 17 \times 13 \times 11 \times 7 \times 5 \times 3 \times 2$?
    # No, it's the part of a specific number.
    # Max $S(m)$ for $m < 10^6$ is bounded.
    # Actually, we can just use a dict.
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Initialize window for n = N_MIN
    # Window covers [N_MIN+1, N_MIN+24]
    freq = {}
    distinct_count = 0
    
    for i in range(1, WINDOW + 1):
        val = extracted[N_MIN + i]
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
            distinct_count += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
        
    # Slide window
    # n goes from N_MIN + 1 to N_MAX
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove n
        val_out = extracted[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            # Optional: del freq[val_out] to keep dict small, but not strictly necessary
            
        # Add n + WINDOW
        val_in = extracted[n + WINDOW]
        if val_in in freq:
            freq[val_in] += 1
        else:
            freq[val_in] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
