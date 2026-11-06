"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # We need to search n in [1000, 1000000]
    # For each n, we look at n+1, n+2, ..., n+24
    # So the numbers we factor range from 1001 to 1000024
    
    # Precompute the small-prime part for all numbers up to 1000024
    # small_part[m] = product of p^k for primes p < 24 dividing m
    # We also track which primes divide each number (as a bitmask)
    
    max_val = 1000024
    small_part = [1] * (max_val + 1)
    prime_mask = [0] * (max_val + 1)  # bitmask of which small primes divide the number
    
    # Sieve-like: for each prime p < 24, multiply into small_part and set mask bit
    for pi, p in enumerate(primes):
        bit = 1 << pi
        for m in range(p, max_val + 1, p):
            # Count power of p in m
            temp = m
            while temp % p == 0:
                small_part[m] *= p
                temp //= p
            prime_mask[m] |= bit
    
    # Now for each window starting at n (n+1 to n+24), compute the OR of prime masks
    # and count distinct bits
    
    # We want to minimize the number of distinct primes across all 24 numbers in the window
    # Use a sliding window with a frequency count of each prime bit
    
    best_n = 1000
    best_count = 99  # worst case is all 9 primes
    
    # Frequency of each prime bit in the current window
    freq = [0] * num_primes
    
    # Initialize window for n=1000: numbers 1001 to 1024
    for i in range(1, 25):
        m = 1000 + i
        mask = prime_mask[m]
        for pi in range(num_primes):
            if mask & (1 << pi):
                freq[pi] += 1
    
    # Count distinct primes in initial window
    cur_distinct = sum(1 for f in freq if f > 0)
    if cur_distinct < best_count:
        best_count = cur_distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Remove n (which was n+0 in previous window, i.e., the number leaving is n)
        # Wait, let me re-think. Window for n is [n+1, n+24].
        # Window for n-1 is [n, n+23].
        # So going from n-1 to n: remove n, add n+24.
        
        # Remove number n (it was in window for n-1 as the first element)
        mask_remove = prime_mask[n]
        for pi in range(num_primes):
            if mask_remove & (1 << pi):
                freq[pi] -= 1
        
        # Add number n+24
        m_add = n + 24
        mask_add = prime_mask[m_add]
        for pi in range(num_primes):
            if mask_add & (1 << pi):
                freq[pi] += 1
        
        cur_distinct = sum(1 for f in freq if f > 0)
        if cur_distinct < best_count:
            best_count = cur_distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
