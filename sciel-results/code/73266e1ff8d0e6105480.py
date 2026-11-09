"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW

    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # s[m] = part of m composed of primes < 24
    s = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Sieve to compute s[m]
    for p in PRIMES:
        for m in range(p, MAX_VAL + 1, p):
            v = m
            while v % p == 0:
                s[m] *= p
                v //= p

    # Sliding window
    freq = {}
    distinct = 0

    # Initialize first window for n=1000: s[1001]..s[1024]
    for i in range(1, WINDOW + 1):
        v = int(s[1000 + i])
        freq[v] = freq.get(v, 0) + 1
        if freq[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct
    best_sum = sum(int(s[1000 + i]) for i in range(1, WINDOW + 1))
    current_sum = best_sum

    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break

        v_out = int(s[n])
        cnt_out = freq[v_out] - 1
        freq[v_out] = cnt_out
        if cnt_out == 0:
            del freq[v_out]
            distinct -= 1

        v_in = int(s[n + WINDOW])
        cnt_in = freq.get(v_in, 0) + 1
        freq[v_in] = cnt_in
        if cnt_in == 1:
            distinct += 1

        current_sum = current_sum - v_out + v_in

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            best_sum = current_sum
            if best_distinct == 1:
                return best_n
        elif distinct == best_distinct and current_sum < best_sum:
            best_n = n
            best_sum = current_sum

    return best_n

# EVOLVE-BLOCK-END
