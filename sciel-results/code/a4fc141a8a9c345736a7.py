"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # up to 1000024
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part[i] = product of p^e for all p in small_primes dividing i
    small_part = [1] * (MAX_VAL + 1)
    for p in small_primes:
        for multiple in range(p, MAX_VAL + 1, p):
            val = multiple
            while val % p == 0:
                small_part[multiple] *= p
                val //= p
    
    # Sliding window to find n minimizing distinct small_part values in [n+1, n+24]
    best_n = 1000
    best_count = float('inf')
    
    from collections import defaultdict
    
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = defaultdict(int)
    distinct = 0
    for i in range(1001, 1001 + WINDOW):
        sp = small_part[i]
        if freq[sp] == 0:
            distinct += 1
        freq[sp] += 1
    
    best_count = distinct
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove n (which is (n-1)+1), add n+WINDOW
        old_val = small_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct -= 1
        
        new_val = small_part[n + WINDOW]
        if freq[new_val] == 0:
            distinct += 1
        freq[new_val] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
