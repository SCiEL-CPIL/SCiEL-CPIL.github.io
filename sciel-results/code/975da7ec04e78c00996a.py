import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill

    MAX_N = 1000000
    MAX_VAL = MAX_N + 24  # we need up to n+24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Compute the 23-smooth part (largest divisor using only primes < 24)
    # for each number 1..MAX_VAL using a sieve approach.
    # smooth_val[i] = product of all prime-power factors of i where prime < 24
    smooth_val = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            smooth_val[power::power] *= p
            power *= p

    # smooth_val[i] now equals the largest divisor of i composed of primes < 24
    # Verify: for i=12=2^2*3, smooth_val[12] should be 12
    # p=2: power=2, multiples of 2 get *2; power=4, multiples of 4 get *2; power=8... 
    # 12 is multiple of 2 -> *2, multiple of 4 -> *2, so factor from 2 is 4. Correct (2^2).
    # p=3: power=3, 12 is multiple of 3 -> *3. So smooth_val[12] = 4*3 = 12. Correct.

    # Now we want for each window [n+1, n+24], the number of distinct smooth_val values.
    # Precompute the smooth_val array for indices 1..MAX_VAL.
    # smooth_val[0] is unused.

    # For efficiency in the sliding window count, use a dictionary-based approach
    # but with early termination when we already have too many distinct values.

    best_n = 1000
    best_count = 25  # worst case is 25 distinct values (all different)

    # Use a frequency array for the sliding window
    # smooth_val values can be up to MAX_VAL, so we use an array of size MAX_VAL+1
    freq = np.zeros(MAX_VAL + 1, dtype=np.int32)
    distinct = 0

    # Initialize window for n=1000: indices 1001..1024
    for idx in range(1001, 1025):
        v = int(smooth_val[idx])
        if freq[v] == 0:
            distinct += 1
        freq[v] += 1

    if distinct < best_count:
        best_count = distinct
        best_n = 1000

    # Slide the window
    for n in range(1001, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Remove index n (which was n-1+1 = n)
        # Window for n-1 was [n, n+23]
        # Window for n is [n+1, n+24]
        # Remove: index n
        # Add: index n+24
        v_out = int(smooth_val[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            distinct -= 1

        v_in = int(smooth_val[n + 24])
        if freq[v_in] == 0:
            distinct += 1
        freq[v_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n

            # If we hit the theoretical minimum, we can stop
            # The minimum possible is 1 (all same smooth value)
            if best_count == 1:
                break

    return best_n
