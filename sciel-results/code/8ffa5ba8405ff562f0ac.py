# EVOLVE-BLOCK-START
import numpy as np

def construct():
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size + 10  # Safety margin

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # small_part[m] = product of p^k for p in small_primes where p^k divides m
    small_part = np.ones(max_val + 1, dtype=np.int64)

    for p in small_primes:
        power = p
        while power <= max_val:
            multiples = np.arange(power, max_val + 1, power)
            small_part[multiples] *= p
            power *= p

    # Sliding window to minimize distinct count of small_part[n+1]...small_part[n+24]
    freq = {}
    current_distinct = 0

    # Initialize window for n=1000 (indices 1001 to 1024)
    for i in range(1001, 1001 + window_size):
        v = small_part[i]
        freq[v] = freq.get(v, 0) + 1
        if freq[v] == 1:
            current_distinct += 1

    best_n = 1000
    best_count = current_distinct

    # Slide window
    for n in range(1001, max_n + 1):
        # Remove small_part[n]
        v_out = small_part[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            current_distinct -= 1

        # Add small_part[n+24]
        v_in = small_part[n + 24]
        freq[v_in] = freq.get(v_in, 0) + 1
        if freq[v_in] == 1:
            current_distinct += 1

        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break

    return best_n

# EVOLVE-BLOCK-END
