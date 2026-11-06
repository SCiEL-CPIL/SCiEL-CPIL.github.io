def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024
    
    # Compute 23-smooth part for all numbers up to max_val
    # smooth_part[m] = the part of m composed only of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            multiples = np.arange(power, max_val + 1, power)
            smooth_part[multiples] *= p
            power *= p
    
    # Extract the relevant range: positions 1001 to 1000024
    # For n in [1000, 1000000], window is n+1 to n+24
    # So we need smooth_part[1001 : 1000025]
    # Array index: smooth_part[1001] corresponds to number 1001
    
    S = smooth_part[1001:1000025]  # length = 1000024 - 1001 + 1 = 1000024
    # Wait, let me recalculate. n ranges from 1000 to 1000000.
    # For n=1000: window is 1001..1024
    # For n=1000000: window is 1000001..1000024
    # So we need smooth_part[1001] through smooth_part[1000024]
    # That's indices 1001 to 1000024 inclusive, length = 1000024 - 1001 + 1 = 1000024
    
    # Actually: 1000024 - 1001 + 1 = 1000024. Yes.
    
    num_windows = 1000000 - 1000 + 1  # = 1000001 windows
    # Window n (0-indexed in our array) covers S[n] to S[n+23]
    # where S[0] = smooth_part[1001], S[1] = smooth_part[1002], etc.
    
    # Vectorized distinct count using first-occurrence method
    # For each window starting at position n (0-indexed in S),
    # count how many elements S[n+j] (j=0..23) are first occurrences
    
    distinct_counts = np.ones(num_windows, dtype=np.int32)  # j=0 always contributes 1
    
    for j in range(1, 24):
        # F_j[n] = 1 if S[n+j] != S[n+k] for all k in 0..j-1
        # S[n+j] is at index n+j in S
        # We need to compare S[n+j] with S[n], S[n+1], ..., S[n+j-1]
        
        # Create array of S[n+j] for all valid n
        # n ranges from 0 to num_windows-1
        # S[n+j] is valid when n+j < len(S), i.e., n < len(S) - j
        # len(S) = 1000024, j up to 23, so n < 1000024 - 23 = 1000001 = num_windows. Good.
        
        elem = S[j:j+num_windows]  # S[n+j] for n=0..num_windows-1
        
        # Check against S[n+k] for k=0..j-1
        is_new = np.ones(num_windows, dtype=bool)
        for k in range(j):
            prev = S[k:k+num_windows]  # S[n+k] for n=0..num_windows-1
            is_new &= (elem != prev)
        
        distinct_counts += is_new.astype(np.int32)
    
    # Find the minimum
    min_count = np.min(distinct_counts)
    min_indices = np.where(distinct_counts == min_count)[0]
    best_n = 1000 + min_indices[0]  # n = 1000 + offset
    
    return int(best_n)
