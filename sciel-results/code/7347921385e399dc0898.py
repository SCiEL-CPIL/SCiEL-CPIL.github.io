import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small prime part using sieve-like approach
    # S[m] = product of p^k where p^k || m for all primes p < 24
    S = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            if time.time() > deadline - 10:
                break
            # For all multiples of pk, multiply S by p
            # This correctly accumulates the full power of p in each number
            S[pk::pk] *= p
            pk *= p
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline - 10:
        # Fallback if precomputation took too long
        return min_n
    
    # Sliding window search with frequency dictionary
    # Window for n: S[n+1], S[n+2], ..., S[n+24]
    
    # Initialize window for n = min_n
    freq = {}
    distinct_count = 0
    
    for i in range(min_n + 1, min_n + 25):
        val = int(S[i])
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
            distinct_count += 1
    
    best_count = distinct_count
    best_n = min_n
    
    if best_count == 1:
        return best_n
    
    # Slide the window
    # For n -> n+1: remove S[n+1], add S[n+25]
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline - 2:
            break
        
        # Remove S[n] (which was the first element of previous window)
        # Previous window for n-1: S[n], S[n+1], ..., S[n+23]
        # New window for n: S[n+1], S[n+2], ..., S[n+24]
        # So remove S[n], add S[n+24]
        
        old_val = int(S[n])
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1
        
        new_val = int(S[n + 24])
        if new_val in freq:
            freq[new_val] += 1
        else:
            freq[new_val] = 1
            distinct_count += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
    
    return best_n
