import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    
    # Initialize F array with 1s
    # Using a list of ints might be faster for Python-level access in the sliding window
    # than a numpy array, because numpy indexing has overhead for single elements.
    # However, computing the sieve is easier with numpy or just a list.
    # Let's use a list for S (small prime parts).
    S = [1] * (max_val + 1)
    
    # Sieve for the extracted factors
    # For each prime p, multiply S[m] by p^v_p(m)
    # We can do this efficiently by iterating multiples.
    for p in primes:
        for m in range(p, max_val + 1, p):
            # Divide m by p as long as possible, multiplying S[m] by p
            x = m
            while x % p == 0:
                S[m] *= p
                x //= p

    # Now S[i] contains the extracted factor for i.
    
    # Sliding window approach
    # Window size K = 24
    K = 24
    
    # Initialize frequency map for the first window: indices 1001 to 1024
    # n starts at 1000, so window is S[1001] ... S[1024]
    freq = {}
    distinct_count = 0
    
    for i in range(1001, 1001 + K):
        val = S[i]
        if val not in freq:
            freq[val] = 1
            distinct_count += 1
        else:
            freq[val] += 1
            
    best_n = 1000
    min_distinct = distinct_count
    
    # Slide the window
    # n goes from 1000 to max_n
    # Current window for n is S[n+1] ... S[n+K]
    # Next window for n+1 is S[n+2] ... S[n+K+1]
    # So we remove S[n+1] and add S[n+K+1]
    
    for n in range(1000, max_n + 1):
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
        # If we reached the theoretical minimum (unlikely to be 1, maybe 2?), we could break, 
        # but let's just continue to be safe or break if 1.
        # Actually, if min_distinct == 1, we can break immediately as it's the absolute minimum.
        if min_distinct == 1:
            break
            
        if n == max_n:
            break
            
        # Slide: remove S[n+1], add S[n+K+1]
        # Note: indices are 1-based in the problem description for n+i, 
        # but our array S is 0-indexed. 
        # The problem says "for each n+i, 1<=i<=24". 
        # So the numbers are n+1, ..., n+24.
        # In our S array, these are S[n+1] ... S[n+24].
        
        # Remove S[n+1]
        val_out = S[n+1]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            distinct_count -= 1
            
        # Add S[n+K+1] which is S[n+25]
        val_in = S[n+K+1]
        if val_in not in freq:
            freq[val_in] = 1
            distinct_count += 1
        else:
            freq[val_in] += 1

    return best_n
