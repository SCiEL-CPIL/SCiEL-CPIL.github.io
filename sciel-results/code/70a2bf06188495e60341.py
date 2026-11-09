"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW

    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # s[m] will store the part of m composed of primes < 24
    s = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Sieve to compute s[m] using vectorized operations
    for p in PRIMES:
        # Find all multiples of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, extract the p-part
        vals = multiples.copy()
        while np.any(vals % p == 0):
            mask = vals % p == 0
            s[multiples[mask]] *= p
            vals[mask] //= p

    # Sliding window to find min distinct count
    # Window covers s[n+1] ... s[n+24]
    # Start with n = 1000
    # Indices: 1001 ... 1024

    freq = {}
    distinct = 0

    # Initialize first window for n=1000
    # s[1001] to s[1024]
    for i in range(1, WINDOW + 1):
        v = int(s[1000 + i])
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1

    best_n = 1000
    best_distinct = distinct
    current_sum = sum(int(s[1000 + i]) for i in range(1, WINDOW + 1))
    best_sum = current_sum

    # Slide window
    # n goes from 1001 to N
    # New element entering: s[n+24]
    # Old element leaving: s[n]

    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break

        # Remove s[n]
        v_out = int(s[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1

        # Add s[n+24]
        v_in = int(s[n + 24])
        if v_in in freq:
            freq[v_in] += 1
        else:
            freq[v_in] = 1
            distinct += 1

        # Update current sum
        current_sum = current_sum - v_out + v_in

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            best_sum = current_sum
            # Early exit if we hit the theoretical minimum?
            # The minimum possible distinct values is 1.
            if best_distinct == 1:
                return best_n
        elif distinct == best_distinct and current_sum < best_sum:
            # Prefer smaller sum of extracted factors as tiebreaker
            best_n = n
            best_sum = current_sum

    return best_n

# EVOLVE-BLOCK-END
