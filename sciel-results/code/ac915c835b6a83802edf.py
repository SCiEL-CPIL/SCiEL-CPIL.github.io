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
    # using vectorized numpy operations
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        # For each power p^k, multiply extracted[m] by p for m divisible by p^k
        # but NOT by p^(k+1)
        p_power = p
        while p_power <= MAX_VAL:
            # Indices divisible by p_power
            idx = np.arange(p_power, MAX_VAL + 1, p_power, dtype=np.int64)
            # Filter: those NOT divisible by p_power * p
            if p_power * p <= MAX_VAL:
                mask = (idx % (p_power * p)) != 0
                idx = idx[mask]
            extracted[idx] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Sliding window using numpy for speed
    # We need to track distinct count in a window of 24 consecutive extracted values
    # Use a frequency dict but with numpy-accelerated checks

    # Precompute the extracted array as a regular list for faster dict access
    # Actually, let's use numpy arrays and a different approach

    # Convert to list for faster scalar access in the sliding window
    ext_list = extracted.tolist()

    from collections import defaultdict

    # Initialize window for n = N_MIN: covers N_MIN+1 to N_MIN+24
    freq = defaultdict(int)
    for i in range(1, WINDOW + 1):
        freq[ext_list[N_MIN + i]] += 1

    distinct_count = len(freq)
    best_n = N_MIN
    best_count = distinct_count

    # Slide the window with periodic time checks
    check_interval = 10000  # check time every 10000 iterations
    for n in range(N_MIN + 1, N_MAX + 1):
        if (n - N_MIN) % check_interval == 0 and time.time() - start_time > DEADLINE:
            break

        # Remove extracted[n], add extracted[n+24]
        old_val = ext_list[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1

        new_val = ext_list[n + 24]
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
