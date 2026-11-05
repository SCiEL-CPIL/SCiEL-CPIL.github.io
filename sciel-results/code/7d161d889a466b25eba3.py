import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # Max value we need to compute: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = [1] * (MAX_VAL + 1)
    
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p
    
    # Sliding window to find n that minimizes distinct extracted factors
    # We track the count of each factor in the current window
    # Since factors can be large, we use a dictionary
    
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Initialize counts for the first window n = N_MIN
    # Window contains n+1 to n+WINDOW
    factor_counts = {}
    distinct_count = 0
    
    for i in range(1, WINDOW + 1):
        f = extracted[N_MIN + i]
        if f not in factor_counts:
            factor_counts[f] = 0
            distinct_count += 1
        factor_counts[f] += 1
    
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
    
    # Slide the window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove n (which is n-1 + 1)
        f_out = extracted[n]
        factor_counts[f_out] -= 1
        if factor_counts[f_out] == 0:
            distinct_count -= 1
            # Optionally delete to save memory, but not strictly necessary
            # del factor_counts[f_out]
        
        # Add n + WINDOW
        f_in = extracted[n + WINDOW]
        if f_in not in factor_counts:
            factor_counts[f_in] = 0
            distinct_count += 1
        factor_counts[f_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
