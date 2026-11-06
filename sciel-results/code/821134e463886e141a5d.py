import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit for safety
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small parts for range [1, 1000024]
    # We need n up to 1000000, so n+24 up to 1000024
    max_val = 1000024
    small_part = [1] * (max_val + 1)
    
    # For each prime, multiply small_part[x] by p^k for all x divisible by p
    # This is more efficient than trial division for each x if we do sieve-like approach
    # But simple loop is fine for 10^6 * 9
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            # Calculate p^k part
            val = multiple
            factor_part = 1
            while val % p == 0:
                factor_part *= p
                val //= p
            small_part[multiple] *= factor_part
            
    best_n = 1000
    min_distinct = float('inf')
    
    # Iterate n from 1000 to 1000000
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
            
        # Get factors for n+1 to n+24
        # Using a set to count distinct values
        # Optimization: if we see many distinct values early, we can break?
        # Max possible distinct is 24.
        # If min_distinct is 2, we can't do better than 2 (since 1 is impossible? 
        # If all factors are 1, count is 1. But 2 divides some, so some factor > 1.
        # So min is at least 2.
        # If we find 2, we can stop?
        # Can we get 1? Only if all 24 numbers have factor 1. Impossible.
        # So 2 is the theoretical minimum.
        
        factors = set()
        # Unroll loop for speed? Or just use set comprehension
        # set(small_part[n+i] for i in range(1, 25))
        
        # Manual loop might be faster than generator expression in tight loop?
        # Let's try standard set comprehension first.
        s = set()
        for i in range(1, 25):
            s.add(small_part[n+i])
            
        count = len(s)
        if count < min_distinct:
            min_distinct = count
            best_n = n
            if min_distinct == 2:
                # 2 is the minimum possible (since 1 is impossible)
                # We can break early
                break
                
    return best_n
