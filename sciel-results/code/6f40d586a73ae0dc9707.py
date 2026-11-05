import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, leaving 10s margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers to consider
    # n in [1000, 1000000]
    # n+i in [1001, 1000024]
    max_n = 1000000
    min_n = 1000
    
    # We need smooth parts for numbers from 1001 to 1000024
    # Let's compute for 1 to 1000024 to be safe, or just the range.
    # Efficient precomputation using sieve-like method?
    # Or just loop?
    # 10^6 numbers * 9 primes is fast enough in Python if vectorized or optimized.
    
    # Let's use a standard loop for smooth part calculation, it's simple.
    # s[m] = m
    # for p in primes:
    #   while s[m] % p == 0: s[m] //= p
    # This is wrong. We want the factor COMPOSED of these primes.
    # So we multiply them back?
    # No, "extract its complete factor composed of primes...".
    # This is the largest divisor d of m such that all prime factors of d are in primes.
    # So:
    # d = 1
    # for p in primes:
    #   while m % p == 0:
    #     d *= p
    #     m //= p
    # return d
    
    # Let's precompute an array `sp` where sp[i] is the smooth part of i.
    # Size 1000025.
    size = 1000025
    sp = np.ones(size, dtype=np.int32)
    
    # Initialize sp[i] = i? No, we build it up.
    # Actually, it's easier to just compute it for each number.
    # But we can do it smarter.
    # For each prime p, for each multiple of p, we can't just multiply because of powers.
    
    # Let's just do the division method in a loop. It might be slow in pure Python for 10^6.
    # 10^6 * 9 * (avg divisions) ~ 10^7 operations. Python handles ~10^7 simple ops in 1s.
    # So ~1-2 seconds. Acceptable.
    
    sp = np.zeros(size, dtype=np.int32)
    for i in range(1, size):
        m = i
        d = 1
        for p in primes:
            while m % p == 0:
                d *= p
                m //= p
        sp[i] = d
        
    # Now find min distinct count in windows of 24
    min_count = 24 # Max possible
    best_n = 1000
    
    # Iterate n from 1000 to 1000000
    # To speed up, we can use a sliding window set?
    # But we need distinct count.
    # A set of 24 elements.
    # Creating a set 10^6 times might be slow.
    # 10^6 set creations of size 24.
    # Let's try it.
    
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
            
        # Get the 24 smooth parts
        window = sp[n+1 : n+25]
        # Count distinct
        # Using set is O(24).
        distinct_count = len(set(window))
        
        if distinct_count < min_count:
            min_count = distinct_count
            best_n = n
            if min_count == 1:
                break # Can't get better than 1
    
    return best_n
