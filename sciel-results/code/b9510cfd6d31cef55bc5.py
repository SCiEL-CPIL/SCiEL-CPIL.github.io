import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max n is 1,000,000. We need F up to 1,000,024.
    max_n = 1000000
    max_val = max_n + 24
    
    # Initialize F array with 1s
    F = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve for the extracted factors
    for p in primes:
        # We need to multiply F[k] by p^v_p(k)
        # It is more efficient to iterate multiples and handle powers
        # Or just:
        # For each multiple m of p:
        #   temp = m
        #   power = 1
        #   while temp % p == 0:
        #       power *= p
        #       temp //= p
        #   F[m] *= power
        # This is O(N log N) worst case but average is low.
        # Alternative:
        # F[m] *= p
        # Then for multiples of p^2:
        # F[m] *= p
        # etc.
        
        # Let's use the power counting method for clarity and safety
        # Actually, standard sieve for "part of number composed of primes"
        # is to just multiply by p for each prime factor.
        
        # Efficient way:
        # Create a copy or just update in place?
        # F is int64.
        
        # Let's do:
        # for multiple in range(p, max_val+1, p):
        #    x = multiple
        #    while x % p == 0:
        #        F[multiple] *= p
        #        x //= p
        
        # This is fine.
        for m in range(p, max_val + 1, p):
            x = m
            while x % p == 0:
                F[m] *= p
                x //= p

    # Now F[i] contains the extracted factor for i.
    
    best_n = 1000
    min_distinct = float('inf')
    
    # We need to check n in [1000, 1000000]
    # For each n, look at F[n+1] ... F[n+24]
    
    # To speed up, we can just loop.
    # 10^6 iterations.
    # Creating a set of 24 numpy integers might be slow in pure python loop.
    # We can use a sliding window approach or just simple loop if fast enough.
    # 10^6 * 24 operations is ~24M operations. Python might take a few seconds.
    # Let's try to optimize the inner loop.
    
    # We can precompute the distinct counts for all windows of size 24?
    # That's hard because "distinct" is not additive.
    
    # Let's just loop and hope it's fast enough.
    # If not, we can use numpy to extract slices and use np.unique.
    # np.unique on an array of 24 elements is fast.
    
    for n in range(1000, max_n + 1):
        window = F[n+1:n+25]
        # Count distinct values
        # np.unique returns sorted unique values
        distinct_count = len(np.unique(window))
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            # Optional: early stop if we reach theoretical minimum?
            # Theoretical minimum is 1 (all F equal). Unlikely.
            # Maybe 2 or 3?
            if min_distinct <= 3:
                # We can probably break or continue to find even lower?
                # Let's continue to be sure, or break if 1?
                # Let's just record the best.
                pass

    return best_n
