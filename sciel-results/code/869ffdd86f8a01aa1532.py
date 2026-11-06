import numpy as np

def construct():
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = 1000024
    min_val = 1001
    
    # Precompute small_prime_part for all numbers in [min_val, max_val]
    part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        start = ((min_val + p - 1) // p) * p
        for m in range(start, max_val + 1, p):
            power = 1
            temp = m
            while temp % p == 0:
                temp //= p
                power *= p
            part[m] *= power
            
    best_n = 1000
    best_count = 25
    
    # Sliding window over n in [1000, 1000000]
    # Window is part[n+1] ... part[n+24]
    
    # Initialize frequency count for the first window n=1000
    # Indices: 1001 to 1024
    freq = {}
    for i in range(1001, 1025):
        val = part[i]
        freq[val] = freq.get(val, 0) + 1
        
    current_distinct = len(freq)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
        
    for n in range(1001, 1000001):
        # Move window from n-1 to n
        # Remove part[n] (first element of previous window)
        # Add part[n+24] (last element of new window)
        
        out_val = part[n]
        in_val = part[n + 24]
        
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            
        freq[in_val] = freq.get(in_val, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break
                
    return best_n
