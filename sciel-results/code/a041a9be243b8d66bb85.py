"""Combined program: Segmented sieve + sliding window for distinct totients."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = b'\x00' * len(range(i*i, limit+1, i))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    small_primes = sieve_primes(100000)

    def phi_trial(x, primes=None):
        """Compute phi(x) using trial division by primes."""
        if primes is None:
            primes = small_primes
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_longest_distinct_run_sliding(phi_vals, offset=0):
        """Find longest run of consecutive distinct values using sliding window.
        Returns (best_k, best_start)."""
        n = len(phi_vals)
        if n == 0:
            return 0, offset

        best_k = 0
        best_start = offset

        # Use a dict to track counts for sliding window
        from collections import defaultdict
        count = defaultdict(int)
        left = 0
        window_distinct = 0

        for right in range(n):
            v = phi_vals[right]
            count[v] += 1
            if count[v] == 1:
                window_distinct += 1

            # Shrink window from left while we have duplicates
            while count[phi_vals[left]] > 1:
                count[phi_vals[left]] -= 1
                if count[phi_vals[left]] == 0:
                    window_distinct -= 1
                left += 1

            # Current window [left, right] has all distinct values
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_start = offset + left

        return best_k, best_start

    def check_window(start, length, primes=None):
        """Check if a window has all distinct phi values, return max distinct prefix length."""
        if primes is None:
            primes = small_primes
        seen = set()
        k = 0
        for i in range(length):
            v = phi_trial(start + i, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def compute_phi_sieve(limit):
        """Compute phi for all numbers up to limit using sieve."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi[j] -= phi[j] // i
        return phi

    def segmented_phi(low, high, primes=None):
        """Compute phi(n) for n in [low, high] using segmented sieve."""
        if primes is None:
            primes = small_primes

        n = high - low + 1
        phi = np.arange(low, high + 1, dtype=np.int64)

        max_p = int(math.isqrt(high)) + 1
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            indices = np.arange(first - low, n, p)
            phi[indices] = phi[indices] - phi[indices] // p

        # Handle remaining prime factors > sqrt(high)
        reduced = np.arange(low, high + 1, dtype=np.int64)
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            indices = np.arange(first - low, n, p)
            for _ in range(40):
                mask = reduced[indices] % p == 0
                if not np.any(mask):
                    break
                reduced[indices[mask]] = reduced[indices[mask]] // p

        mask = reduced > 1
        if np.any(mask):
            phi[mask] = phi[mask] - phi[mask] // reduced[mask]

        return phi

    best_k = 1
    best_start = 1

    # Phase 1: Sieve up to 10^7 for exhaustive small-range search
    if time_remaining() > 30.0:
        sieve_limit = 10**7
        try:
            phi_vals = compute_phi_sieve(sieve_limit)
            k, s = find_longest_distinct_run_sliding(phi_vals[1:], offset=1)
            if k > best_k:
                best_k = min(k, 200)
                best_start = s
        except MemoryError:
            pass
        except Exception:
            pass

    # Phase 2: Segmented sieve over larger ranges
    CHUNK_SIZE = 200000
    search_ranges = [
        (10**11, 10**12),
        (5*10**10, 10**11),
        (10**10, 5*10**10),
        (10**9, 10**10),
        (10**8, 10**9),
        (10**7, 10**8),
    ]

    for range_start, range_end in search_ranges:
        if time_remaining() < 15.0:
            break
        search_pos = range_start
        while search_pos < range_end and time_remaining() > 8.0:
            chunk_end = min(search_pos + CHUNK_SIZE - 1, range_end, 10**12)
            try:
                phi_vals = segmented_phi(search_pos, chunk_end)
                k, s = find_longest_distinct_run_sliding(phi_vals, offset=search_pos)
                if k > best_k:
                    best_k = min(k, 200)
                    best_start = s
                    if best_k >= 200:
                        return {'start': best_start, 'length': best_k}
            except MemoryError:
                continue
            except Exception:
                pass
            search_pos = chunk_end + 1

    # Phase 3: Targeted search around promising locations
    if time_remaining() > 10.0 and best_k < 200:
        random.seed(12345)

        # Search around primorial multiples and other structured locations
        primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]

        for prim in primorials:
            if time_remaining() < 5.0:
                break
            max_mult = min(10**12 // prim, 20000)
            for _ in range(min(30, max_mult)):
                if time_remaining() < 3.0:
                    break
                mult = random.randint(1, max_mult)
                start = mult * prim
                if start < 1 or start + 200 > 10**12:
                    continue
                k = check_window(start, 200)
                if k > best_k:
                    best_k = k
                    best_start = start
                for offset in [-20, -10, -5, -1, 1, 5, 10, 20]:
                    s2 = start + offset
                    if s2 < 1 or s2 + 200 > 10**12:
                        continue
                    k2 = check_window(s2, 200)
                    if k2 > best_k:
                        best_k = k2
                        best_start = s2

    # Phase 4: Random search in high ranges
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(150):
            if time_remaining() < 2.0:
                break
            r = random.random()
            if r < 0.4:
                s = random.randint(10**11, 10**12 - 200)
            elif r < 0.7:
                s = random.randint(10**10, 10**11 - 200)
            elif r < 0.9:
                s = random.randint(10**9, 10**10 - 200)
            else:
                s = random.randint(10**7, 10**9 - 200)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 5: Local refinement
    if best_k > 1 and time_remaining() > 3.0:
        for delta in range(-100, 101):
            if time_remaining() < 1.0:
                break
            s = best_start + delta
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
