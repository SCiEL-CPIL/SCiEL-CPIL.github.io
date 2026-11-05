"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    N_MAX = 1000000
    WINDOW = 24
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute the smooth part (part composed of primes < 24) for each number
    # smooth_part[x] = the part of x made up of primes < 24
    # i.e., divide out all factors of primes >= 25
    # Actually: extract the part composed of primes strictly below 24
    # This means: for each n+i, take the product of p^a where p < 24 and a >= 1
    
    # To compute efficiently: for each number, remove all prime factors >= 25
    # The remaining part is the "extracted factor"
    
    # Precompute smallest prime factor (SPF) up to N_MAX + WINDOW
    limit = N_MAX + WINDOW
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each number, compute its part composed of primes < 24
    # A prime p is "small" if p < 24
    small_prime_set = set(small_primes)
    
    smooth_part = [1] * (limit + 1)
    for x in range(1, limit + 1):
        n = x
        result = 1
        while n > 1:
            p = spf[n]
            if p in small_prime_set:
                # This prime is < 24, include it
                while n % p == 0:
                    result *= p
                    n //= p
            else:
                # This prime is >= 24, skip it (don't include)
                while n % p == 0:
                    n //= p
        smooth_part[x] = result
    
    # Now slide a window of size 24 over n from 1000 to N_MAX
    # For window starting at n, we look at n+1, n+2, ..., n+24
    # Count distinct smooth_part values in that window
    
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: elements are 1001..1024
    window_vals = [smooth_part[1000 + i] for i in range(1, WINDOW + 1)]
    freq = Counter(window_vals)
    best_count = len(freq)
    best_n = 1000
    
    # Slide the window
    for n in range(1001, N_MAX + 1):
        if time.time() > deadline:
            break
        # Remove n (which was n+0 in previous window, now out)
        # Previous window: n-1+1 .. n-1+24 = n .. n+23
        # New window: n+1 .. n+24
        # Remove smooth_part[n], add smooth_part[n+24]
        old_val = smooth_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = smooth_part[n + WINDOW]
        freq[new_val] += 1
        
        current_distinct = len(freq)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
