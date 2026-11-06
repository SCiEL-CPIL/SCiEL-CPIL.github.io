import time

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_PRODUCT = 223092870  # 2*3*5*7*11*13*17*19*23

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    lower = 1000
    upper = 1000000
    max_val = upper + 24  # 1000024

    # Sieve: mark numbers NOT coprime to _PRODUCT in [lower, max_val]
    # is_coprime[v - lower] = True if gcd(v, _PRODUCT) == 1
    size = max_val - lower + 1
    is_coprime = bytearray(b'\x01') * size  # all True initially

    for p in _SMALL_PRIMES:
        # Mark multiples of p in [lower, max_val] as not coprime
        # First multiple of p >= lower
        first = ((lower + p - 1) // p) * p
        if first < lower:
            first += p
        for m in range(first, max_val + 1, p):
            is_coprime[m - lower] = 0

    # Now scan for the best window: 24 consecutive values n+1..n+24
    # We want to minimize distinct small-prime parts.
    # A window where all 24 are coprime gives distinct count = 1 (just {1}).
    # Otherwise, we need to compute actual small-prime parts.

    # First pass: look for a window where all 24 are coprime
    # Window for n: indices (n+1-lower) through (n+24-lower)
    # i.e., is_coprime[1] through is_coprime[24] for n=lower

    # Use a running count of non-coprime in window of size 24
    # Window for n starts at index 1 (value n+1) and ends at index 24 (value n+24)
    # For n: indices in is_coprime: (n+1-lower) to (n+24-lower)

    # Count non-coprime in initial window (n=lower): indices 1..24
    non_coprime_count = 0
    for i in range(1, 25):
        if not is_coprime[i]:
            non_coprime_count += 1

    best_n = lower
    best_count = 25  # worst case: 24 distinct non-1 factors + possibly 1

    if non_coprime_count == 0:
        # All 24 are coprime, so small-prime part is 1 for all
        # Distinct factors = {1}, count = 1
        return lower

    # Slide window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        # Remove index (n-lower) [value n], add index (n+24-lower) [value n+24]
        out_idx = n - lower
        in_idx = n + 24 - lower

        if not is_coprime[out_idx]:
            non_coprime_count -= 1
        if not is_coprime[in_idx]:
            non_coprime_count += 1

        if non_coprime_count == 0:
            return n  # All 24 coprime, distinct count = 1

    # If no all-coprime window found, compute actual small-prime parts
    # for the best candidate windows
    # This shouldn't happen given max gap is 102 > 24, but as fallback:
    # Compute small-prime parts for all numbers and find best window
    sp_parts = [1] * size
    for p in _SMALL_PRIMES:
        first = ((lower + p - 1) // p) * p
        if first < lower:
            first += p
        for m in range(first, max_val + 1, p):
            idx = m - lower
            val = m
            while val % p == 0:
                val //= p
                sp_parts[idx] *= p

    # Sliding window for distinct count
    from collections import defaultdict
    window_counts = defaultdict(int)
    for i in range(1, 25):
        window_counts[sp_parts[i]] += 1
    best_count = len(window_counts)
    best_n = lower

    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
        out_val = sp_parts[n - lower]
        in_val = sp_parts[n + 24 - lower]
        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1
        cur = len(window_counts)
        if cur < best_count:
            best_count = cur
            best_n = n
            if best_count == 1:
                break

    return best_n
