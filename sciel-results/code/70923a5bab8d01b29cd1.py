import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute sf[m] = product of p^k for primes p<24, p^k || m
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, multiply sf[j] by p for each factor of p
        for j in range(p, max_val + 1, p):
            temp = j
            while temp % p == 0:
                sf[j] *= p
                temp //= p
        if time.time() > deadline - 10:
            break
    
    # Sliding window to count distinct values in sf[n+1 : n+25]
    # We use a frequency array indexed by the value of sf[m]
    # Max value of sf[m] is max_val
    freq = np.zeros(max_val + 1, dtype=np.int32)
    distinct_count = 0
    
    # Initialize window for n = min_n
    # The window covers indices min_n+1 to min_n+24
    for i in range(min_n + 1, min_n + 25):
        val = sf[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
        
    best_count = distinct_count
    best_n = min_n
    
    if best_count == 1:
        return best_n
        
    # Slide the window
    for n in range(min_n + 1, max_n + 1):
        # Remove the element that is leaving the window: sf[n]
        # The previous window was [n, n+23] for n-1? 
        # Wait.
        # Window for n is [n+1, n+24].
        # Window for n-1 was [n, n+23].
        # So we remove sf[n] and add sf[n+24].
        
        val_out = sf[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            
        val_in = sf[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
                
        if time.time() > deadline - 5:
            break
            
    return best_n
