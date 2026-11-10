import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 118.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Sieve primes up to 10^6 (sqrt of 10^12)
    limit = 10**6 + 1
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    best_k = 1
    best_s = 1

    max_val = 10**12

    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented approach."""
        n = hi - lo + 1
        if n <= 0:
            return []

        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))

        max_p = int(math.isqrt(hi)) + 1
        # Process primes up to max_p
        for p in primes:
            if p > max_p:
                break
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            # Process multiples
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
        """Find longest run of distinct phi values using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0

        best_k_local = 0
        best_s_local = offset
        last_seen = {}
        start_idx = 0

        for i in range(n):
            v = phi_arr[i]
            if v in last_seen:
                if last_seen[v] >= start_idx:
                    start_idx = last_seen[v] + 1
            last_seen[v] = i
            k = i - start_idx + 1
            if k > best_k_local:
                best_k_local = k
                best_s_local = offset + start_idx
                if best_k_local >= max_len:
                    break

        return best_s_local, best_k_local

    # Generate strategic search ranges
    # Higher values tend to have more varied totients
    # Focus on ranges near primorial multiples for more varied factor structures
    search_ranges = []

    # Primorial-based ranges (numbers near these have many small prime factors)
    # Primorials: 2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]

    # For each primorial, consider ranges near multiples in the high range
    for prim in primorials:
        # Find multiples near high values
        if prim > max_val:
            continue
        # Start from a high multiple and go down
        max_mult = max_val // prim
        # Consider a few multiples
        for m in range(max(1, max_mult - 5), max(1, max_mult) + 1):
            center = m * prim
            # Range of size ~100000 around center
            lo = max(1, center - 50000)
            hi = min(max_val, center + 50000)
            if lo < hi and hi - lo + 1 >= 200:
                search_ranges.append((lo, hi))

    # Also add some high-value ranges not near primorials
    step = 5 * 10**10
    for i in range(20):
        hi = max_val - i * step
        lo = max(1, hi - 100000)
        if lo < hi and hi - lo + 1 >= 200:
            search_ranges.append((lo, hi))

    # Deduplicate
    seen = set()
    unique_ranges = []
    for lo, hi in search_ranges:
        key = (lo, hi)
        if key not in seen:
            seen.add(key)
            unique_ranges.append((lo, hi))

    # Sort by priority: higher start values first, then larger ranges
    unique_ranges.sort(key=lambda x: (-x[0], x[1] - x[0]), reverse=False)
    unique_ranges.sort(key=lambda x: -x[0])

    # Process ranges
    for lo, hi in unique_ranges:
        if time_left() < 15.0:
            break

        # Adaptive window size based on remaining time
        if time_left() > 60.0:
            window = 200000
        elif time_left() > 30.0:
            window = 100000
        else:
            window = 50000

        current_lo = lo
        while current_lo <= hi:
            if time_left() < 10.0:
                break

            current_hi = min(current_lo + window - 1, hi)
            n = current_hi - current_lo + 1

            if n < 200:
                break

            try:
                phi_arr = compute_phi_segmented(current_lo, current_hi)
            except Exception:
                break

            if time_left() < 8.0:
                break

            s, k = find_longest_distinct_run(phi_arr, current_lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}

            current_lo = current_hi + 1

    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > max_val:
        best_k = min(best_k, max_val - best_s + 1)
    if best_k < 1:
        best_k = 1

    return {'start': best_s, 'length': best_k}
