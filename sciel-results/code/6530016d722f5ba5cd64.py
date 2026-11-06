"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute smooth parts
    # smooth_part[m] = factor of m composed of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            m = multiple
            while m % p == 0:
                smooth_part[multiple] *= p
                m //= p
                
    # Convert to list for faster access in Python loop if needed, 
    # but numpy slice + set is also good. 
    # Let's use numpy array directly.
    
    best_n = min_n
    best_score = 25  # Max possible distinct is 24, but 25 is safe init
    
    # Search
    # We can optimize by checking time every N steps
    n = min_n
    while n <= max_n:
        if time.time() > deadline:
            break
        
        # Get the 24 factors
        # n+1 to n+24
        # Indices: n+1 ... n+24
        window = smooth_part[n+1 : n+25]
        
        # Count distinct
        # Using set is efficient for small arrays
        distinct = len(set(window))
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            # Heuristic: If we find a very low number, it's likely optimal
            if best_score <= 3:
                # We can break early with high confidence, 
                # but to be safe, let's run a bit more or just break?
                # Given the constraint, breaking on 3 is a good bet for speed,
                # but might miss a 2. 
                # However, 2 is extremely unlikely for 24 consecutive numbers.
                # 3 is very unlikely.
                # If we find 3, we can break.
                pass # Don't break yet, let's see if we can find 2? 
                # Actually, let's just break if we find 1 or 2. 
                # For 3, we continue to be safe, but it's fast enough.
        
        n += 1
        
    return best_n
# EVOLVE-BLOCK-END
