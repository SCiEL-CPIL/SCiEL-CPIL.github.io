"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = N_MAX + WINDOW  # we need small_part up to N_MAX + 24
    
    # Sieve to compute small_part[m] = part of m with only prime factors < 24
    small_part = [1] * (max_val + 1)
    
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            temp = m
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 0:
                small_part[m] *= (p ** count)
    
    # Sliding window to find n minimizing distinct count of small_part[n+1..n+24]
    best_count = float('inf')
    best_n = N_MIN
    
    freq = defaultdict(int)
    
    # Initial window for n = N_MIN: indices N_MIN+1 to N_MIN+24
    for i in range(N_MIN + 1, N_MIN + 1 + WINDOW):
        val = small_part[i]
        freq[val] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
    
    # Slide: for n from N_MIN+1 to N_MAX
    # Window for n: [n+1, n+24]
    # Window for n-1: [n, n+23]
    # Remove: small_part[n], Add: small_part[n+24]
    for n in range(N_MIN + 1, N_MAX + 1):
        # Remove leftmost of previous window: index n
        old_val = small_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add rightmost of new window: index n + 24
        new_val = small_part[n + WINDOW]
        freq[new_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
        
        # Check deadline periodically
        if (n - N_MIN) % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n

# EVOLVE-BLOCK-END
