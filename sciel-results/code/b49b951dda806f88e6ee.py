import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int32)
    
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # Map extracted values to small integer IDs
    # We only care about values from index 1 to MAX_VAL
    # np.unique returns sorted unique values and inverse indices
    unique_vals, inverse_indices = np.unique(extracted[1:], return_inverse=True)
    
    # inverse_indices[i] corresponds to extracted[i+1]
    # So id_array[m] = inverse_indices[m-1] for m >= 1
    # Let's create a 1-based id array for easier indexing
    # id_array[0] is unused
    id_array = np.empty(MAX_VAL + 1, dtype=np.int32)
    id_array[0] = -1 # unused
    id_array[1:] = inverse_indices
    
    K = len(unique_vals)
    
    # Sliding window logic
    # Window for n is extracted[n+1 : n+1+WINDOW]
    # IDs are id_array[n+1 : n+1+WINDOW]
    
    # Initialize counts
    counts = np.zeros(K, dtype=np.int32)
    
    # Initialize window for n = N_MIN
    # Indices: N_MIN+1 to N_MIN+WINDOW
    start_idx = N_MIN + 1
    end_idx = N_MIN + WINDOW + 1 # exclusive
    
    current_distinct = 0
    for i in range(start_idx, end_idx):
        idx = id_array[i]
        counts[idx] += 1
        if counts[idx] == 1:
            current_distinct += 1
            
    best_n = N_MIN
    best_count = current_distinct
    
    # Slide window
    # Next n is N_MIN + 1
    # Remove id_array[N_MIN+1], Add id_array[N_MIN+WINDOW+1]
    # General step for n -> n+1:
    # Remove id_array[n+1]
    # Add id_array[n+WINDOW+1]
    
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove element leaving the window: n (which is (n-1)+1)
        # Wait, window for n-1 is [n, ..., n+23]
        # Window for n is [n+1, ..., n+24]
        # So we remove extracted[n] and add extracted[n+24]
        # In id_array terms:
        # Remove id_array[n]
        # Add id_array[n+24]
        
        rem_id = id_array[n]
        add_id = id_array[n + 24]
        
        counts[rem_id] -= 1
        if counts[rem_id] == 0:
            current_distinct -= 1
            
        counts[add_id] += 1
        if counts[add_id] == 1:
            current_distinct += 1
            
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break

    return best_n
