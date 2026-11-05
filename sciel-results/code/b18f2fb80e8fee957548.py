import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = [1] * (MAX_VAL + 1)

    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # Build a mapping from extracted value to a small integer ID for fast counting
    # This avoids dict-with-large-int overhead in the sliding window
    # First, collect all distinct extracted values
    # Actually, let's just use the dict approach but optimize the sliding window

    # Sliding window state
    # We'll maintain factor_counts and distinct_count
    # The key insight: use random search + sliding window to jump between candidates

    # Initialize at N_MIN
    factor_counts = {}
    distinct_count = 0

    for i in range(1, WINDOW + 1):
        f = extracted[N_MIN + i]
        if f not in factor_counts:
            factor_counts[f] = 0
            distinct_count += 1
        factor_counts[f] += 1

    best_n = N_MIN
    best_count = distinct_count

    if best_count == 1:
        return best_n

    # Now use random search: jump to random positions and score them
    # Using sliding window transitions is O(distance) which is too slow for large jumps
    # Instead, for each random candidate, compute its window score directly
    # but use a fast method: since WINDOW=24 is small, direct computation is fine

    # Strategy: sample many random candidates, compute their distinct count directly
    # Direct computation for 24 elements is very fast (~24 dict lookups)

    # Use a set for direct computation
    rng = random.Random(42)  # deterministic for reproducibility

    # First, do a quick scan of a few thousand consecutive positions to find a good starting point
    # This helps because good candidates may be clustered
    scan_limit = min(N_MIN + 5000, N_MAX)
    for n in range(N_MIN + 1, scan_limit + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Remove n (the element leaving: n+0 = n, but window is n+1..n+24)
        # Actually for sliding: window at n-1 is (n, n+1, ..., n+23)
        # Window at n is (n+1, n+2, ..., n+24)
        # So remove extracted[n], add extracted[n+24]
        f_out = extracted[n]
        factor_counts[f_out] -= 1
        if factor_counts[f_out] == 0:
            distinct_count -= 1

        f_in = extracted[n + WINDOW]
        if f_in not in factor_counts:
            factor_counts[f_in] = 0
            distinct_count += 1
        factor_counts[f_in] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n

    # Now random search
    # For each random candidate, compute distinct count directly (24 lookups)
    # This is O(24) per candidate, very fast
    # We can do ~100k-500k candidates in the remaining time

    attempts = 0
    while time.time() - start_time < DEADLINE - 5.0:
        n = rng.randint(N_MIN, N_MAX)
        attempts += 1

        # Compute distinct count for window n+1 to n+24
        # Use a small set
        seen = set()
        for i in range(1, WINDOW + 1):
            seen.add(extracted[n + i])

        count = len(seen)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                return best_n

        # Periodically check time
        if attempts % 10000 == 0 and time.time() - start_time > DEADLINE - 10.0:
            break

    return best_n
