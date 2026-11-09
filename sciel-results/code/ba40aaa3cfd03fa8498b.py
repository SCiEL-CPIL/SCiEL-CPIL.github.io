"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW

    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    NUM_PRIMES = len(PRIMES)

    # For each prime, assign a bit position
    # bitmask[m] will have bit i set if prime[i] divides m
    bitmask = np.zeros(MAX_VAL + 1, dtype=np.uint64)

    # Sieve to compute bitmask[m]
    for idx, p in enumerate(PRIMES):
        bit = 1 << idx
        for m in range(p, MAX_VAL + 1, p):
            bitmask[m] |= bit

    # Sliding window: maintain count of how many numbers in window have each prime
    # prime_count[j] = number of numbers in current window where prime[j] is present
    prime_count = [0] * NUM_PRIMES
    distinct = 0

    # Initialize first window for n=1000: indices 1001..1024
    for i in range(1, WINDOW + 1):
        bm = bitmask[1000 + i]
        for j in range(NUM_PRIMES):
            if (bm >> j) & 1:
                prime_count[j] += 1
                if prime_count[j] == 1:
                    distinct += 1

    best_n = 1000
    best_distinct = distinct

    # Slide window
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break

        # Remove bitmask[n] (the element leaving the window)
        bm_out = bitmask[n]
        for j in range(NUM_PRIMES):
            if (bm_out >> j) & 1:
                prime_count[j] -= 1
                if prime_count[j] == 0:
                    distinct -= 1

        # Add bitmask[n+24] (the element entering the window)
        bm_in = bitmask[n + 24]
        for j in range(NUM_PRIMES):
            if (bm_in >> j) & 1:
                prime_count[j] += 1
                if prime_count[j] == 1:
                    distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                return best_n

    return best_n

# EVOLVE-BLOCK-END
