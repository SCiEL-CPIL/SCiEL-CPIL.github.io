import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m] = product of p^k for p<24 where p^k || m
    # Using sieve: for each prime p, for each power p^k, multiply all multiples of p^k by p
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Sliding window approach to find min distinct extracted factors
    from collections import defaultdict

    # Initialize window for n = N_MIN: covers N_MIN+1 to N_MIN+24
    freq = defaultdict(int)
    for i in range(1, WINDOW + 1):
        freq[extracted[N_MIN + i]] += 1

    distinct_count = len(freq)
    best_n = N_MIN
    best_count = distinct_count

    # Slide the window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Remove n (which was n-1+1 = n) from the window
        # Window for n-1: (n-1)+1 to (n-1)+24 = n to n+23
        # Window for n: n+1 to n+24
        # So we remove extracted[n] and add extracted[n+24]

        old_val = extracted[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1

        new_val = extracted[n + 24]
        if new_val not in freq:
            freq[new_val] = 1
            distinct_count += 1
        else:
            freq[new_val] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break

    return best_n
