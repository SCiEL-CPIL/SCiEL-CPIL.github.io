import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max n is 1,000,000. We need factors for n+24.
    # So max value is 1,000,024.
    max_val = 1000024
    
    # small_part[i] will hold the part of i composed of primes in 'primes'
    small_part = [1] * (max_val + 1)
    
    # Sieve-like calculation
    for p in primes:
        p_power = p
        while p_power <= max_val:
            for j in range(p_power, max_val + 1, p_power):
                small_part[j] *= p
            p_power *= p
            
    # Search for n in [1000, 1000000]
    best_n = 1000
    best_count = float('inf')
    
    # Optimization: We can break early if we find count 1
    # But let's just find the global min.
    
    # To speed up, we can skip some n? No, just iterate.
    # 1 million iterations is fast.
    
    for n in range(1000, 1000001):
        # Extract factors for n+1 to n+24
        # Using a set to count distinct values
        # small_part[n+1] ... small_part[n+24]
        
        # Check time
        if time.time() > deadline:
            break
            
        # Small optimization: if best_count is 1, we can't do better
        if best_count == 1:
            break
            
        # Create set of the 24 values
        # Slicing and set conversion
        vals = small_part[n+1:n+25]
        distinct_count = len(set(vals))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
