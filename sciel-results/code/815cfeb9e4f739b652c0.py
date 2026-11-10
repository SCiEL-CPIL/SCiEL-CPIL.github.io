"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] correctly using segmented sieve."""
        n = hi - lo + 1
        if n <= 0:
            return []
        if lo == 1:
            phi = [1]
            lo = 2
            n = hi - lo + 1
            if n <= 0:
                return phi

        limit = int(math.isqrt(hi)) + 1

        # Sieve primes up to limit
        is_prime = [True] * (limit + 1)
        primes = []
        if limit >= 2:
            is_prime[0] = is_prime[1] = False
            for i in range(2, limit + 1):
                if is_prime[i]:
                    primes.append(i)
                    if i * i <= limit:
                        for j in range(i * i, limit + 1, i):
                            is_prime[j] = False

        # phi[i] stores phi(lo + i), remaining[i] stores the remaining factor
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))

        for p in primes:
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p

        # Handle remaining prime factors
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]

        return phi

    def find_longest_distinct_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 1

        best_k = 1
        best_s = offset

        # Use a sliding window approach
        left = 0
        count = {}

        for right in range(n):
            val = phi_arr[right]
            count[val] = count.get(val, 0) + 1

            # Shrink window from left if we have duplicates
            while count[val] > 1:
                left_val = phi_arr[left]
                count[left_val] -= 1
                if count[left_val] == 0:
                    del count[left_val]
                left += 1

            # Current window [left, right] has all distinct values
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = offset + left

            # Cap the window size
            if best_k >= max_len:
                break

        # Also check if we can extend beyond max_len (shouldn't happen but be safe)
        if best_k > max_len:
            best_k = max_len
            best_s = offset + (right - best_k + 1)

        return best_s, best_k

    best_s, best_k = 1, 1

    # Strategy: Test ranges at increasing scales
    # For large n, phi values tend to be more distinct
    # Focus on ranges where the density of collisions is lower

    # Test a variety of ranges, prioritizing larger ones
    test_ranges = [
        (10**11, 10**11 + 500000),      # Very large range
        (10**10, 10**10 + 500000),      # Large range
        (10**9, 10**9 + 500000),        # Medium-large
        (10**8, 10**8 + 500000),        # Medium
        (10**7, 10**7 + 500000),        # Small-medium
        (10**6, 10**6 + 500000),        # Small
        (10**5, 10**5 + 500000),        # Small
        (10**4, 10**4 + 500000),        # Very small
        (1, 500000),                    # Start from 1
    ]

    # Also test some random-ish ranges at different scales
    extra_ranges = [
        (3 * 10**11, 3 * 10**11 + 300000),
        (7 * 10**10, 7 * 10**10 + 300000),
        (2 * 10**9, 2 * 10**9 + 300000),
        (5 * 10**8, 5 * 10**8 + 300000),
    ]

    all_ranges = test_ranges + extra_ranges

    for lo, hi in all_ranges:
        if time_left() < 8.0:
            break

        # Ensure hi doesn't exceed 10^12
        if hi > 10**12:
            hi = 10**12
        if lo > hi:
            continue

        try:
            phi_arr = compute_phi_segmented(lo, hi)
            if time_left() < 5.0:
                break

            s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s

            # If we found a very long run, we can stop early
            if best_k >= 150:
                break
        except Exception:
            continue

    # If we haven't found a good run, do a dense search in a smaller range
    if best_k < 30 and time_left() > 15.0:
        try:
            phi_arr = compute_phi_segmented(1, 1000000)
            s, k = find_longest_distinct_run(phi_arr, 1, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            pass

    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
