"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    MAX_VAL = N + WINDOW  # 1000024

    # Precompute small_part[m]: the largest divisor of m composed of primes in PRIMES
    # small_part[m] = product of p^k for all p in PRIMES where p^k divides m
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # For each prime p, multiply small_part[m] by p for every power p^k dividing m
    # This is done by iterating over powers p, p^2, p^3, ...
    # For a given power p^k, all multiples m of p^k get a factor of p multiplied.
    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            # For all m that are multiples of 'power', multiply small_part[m] by p
            # Using numpy slicing for speed
            small_part[power::power] *= p
            power *= p
            if power > MAX_VAL:
                break

    # Now small_part[i] contains the required factor for i.
    # We need to find n in [1000, 1000000] minimizing distinct values in small_part[n+1 : n+25]
    
    # Sliding window frequency count
    # Since small_part values can be large, we use a dictionary for frequency.
    # However, the number of distinct values in a window of 24 is at most 24.
    # A dictionary is fine.
    
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    # Note: n=1000 corresponds to numbers 1001, ..., 1024
    for i in range(1001, 1001 + WINDOW):
        v = small_part[i]
        freq[v] += 1
        if freq[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    # Slide the window
    # n goes from 1001 to 1000000
    # Window for n is [n+1, n+24]
    # When n increases by 1, we remove small_part[n+1] and add small_part[n+25]
    # Wait, for n=1000, window is [1001, 1024].
    # For n=1001, window is [1002, 1025].
    # Remove: small_part[1001]. Add: small_part[1025].
    # Generally: Remove small_part[n], Add small_part[n+24] when moving from n-1 to n?
    # Let's check indices carefully.
    # Window for n: W_n = {n+1, ..., n+24}
    # Window for n+1: W_{n+1} = {n+2, ..., n+25}
    # Element leaving: n+1
    # Element entering: n+25
    
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        
        # Remove element at index n (which was the first element of the previous window n-1? No.)
        # Previous window was for n-1: {n, ..., n+23}
        # Current window is for n: {n+1, ..., n+24}
        # So we remove small_part[n] and add small_part[n+24]?
        # Let's re-verify.
        # n=1000: {1001, ..., 1024}
        # n=1001: {1002, ..., 1025}
        # Removed: 1001. Added: 1025.
        # Formula: Remove index = n. Add index = n + 24.
        
        v_remove = small_part[n]
        freq[v_remove] -= 1
        if freq[v_remove] == 0:
            distinct -= 1

        v_add = small_part[n + 24]
        freq[v_add] += 1
        if freq[v_add] == 1:
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
