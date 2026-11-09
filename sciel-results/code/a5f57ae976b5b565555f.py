import numpy as np
from collections import Counter

def construct():
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW_SIZE = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to consider numbers from N_MIN+1 to N_MAX+24
    M_start = N_MIN + 1
    M_end = N_MAX + WINDOW_SIZE
    size = M_end - M_start + 1
    
    # Precompute S(m): the complete factor of m composed of primes strictly below 24
    S = np.ones(size, dtype=np.int64)
    
    for p in small_primes:
        # Find indices j where M_start + j is divisible by p
        # M_start + j ≡ 0 (mod p) => j ≡ -M_start (mod p)
        first_j = (-M_start) % p
        # Ensure first_j is in [0, p)
        if first_j < 0:
            first_j += p
        
        for j in range(first_j, size, p):
            val = M_start + j
            while val % p == 0:
                S[j] *= p
                val //= p
    
    # Sliding window over S[0..WINDOW_SIZE-1], S[1..WINDOW_SIZE], etc.
    # Window i covers S[i : i+WINDOW_SIZE], corresponding to n = N_MIN + i
    # Wait: if window starts at index i, it covers S[i], S[i+1], ..., S[i+23]
    # S[i] corresponds to M_start + i = N_MIN + 1 + i
    # So window i corresponds to n = N_MIN + i, and the numbers are n+1, n+2, ..., n+24
    
    # Use a dictionary for frequency counting
    freq = Counter()
    distinct_count = 0
    
    # Fill initial window [0, WINDOW_SIZE)
    for k in range(WINDOW_SIZE):
        val = S[k]
        freq[val] += 1
        if freq[val] == 1:
            distinct_count += 1
    
    min_distinct = distinct_count
    best_n = N_MIN
    
    # Slide window from i=1 to i=N_MAX-N_MIN
    for i in range(1, N_MAX - N_MIN + 1):
        # Remove S[i-1]
        remove_val = S[i-1]
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            distinct_count -= 1
        
        # Add S[i+WINDOW_SIZE-1]
        add_val = S[i+WINDOW_SIZE-1]
        freq[add_val] += 1
        if freq[add_val] == 1:
            distinct_count += 1
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = N_MIN + i
            
            if min_distinct == 1:
                break
    
    return best_n
