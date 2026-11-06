"""Improved program: Fast segmented sieve for long runs of distinct totients."""
import time
import math
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes for trial division and segmented sieve
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=np.uint8)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = 0
        return np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int64)

    small_primes = sieve_primes(100000)  # primes up to 100000

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

    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive distinct values in phi_vals.
        Returns (best_k, best_start) where best_start is the actual number (offset + index)."""
        n = len(phi_vals)
        best_k = 0
        best_start = offset
        seen = set()
        run_start = 0
        run_len = 0

        for i in range(n):
            v = phi_vals[i]
            if v not in seen:
                seen.add(v)
                run_len += 1
            else:
                if run_len > best_k:
                    best_k = run_len
                    best_start = offset + run_start
                seen = {v}
                run_start = i
                run_len = 1

        if run_len > best_k:
            best_k = run_len
            best_start = offset + run_start

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

    def segmented_phi_fast(low, high, primes=None):
        """Compute phi(n) for all n in [low, high] using fast segmented sieve.
        Returns numpy array of phi values."""
        if primes is None:
            primes = small_primes
        
        n = high - low + 1
        # Initialize phi array with the numbers themselves
        phi = np.arange(low, high + 1, dtype=np.int64)
        
        # For each prime p <= sqrt(high), apply the (1 - 1/p) factor to multiples of p
        max_p = int(math.isqrt(high)) + 1
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            # Find first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            # Apply phi[j] -= phi[j] // p for all multiples of p in range
            indices = np.arange(first - low, n, p)
            phi[indices] = phi[indices] - phi[indices] // p
        
        # For the remaining prime factor: after dividing by all primes <= sqrt(high),
        # if the original number had a prime factor > sqrt(high), it must be exactly one
        # (since two such factors would exceed high). We need to detect and handle this.
        # 
        # Key insight: For a number m in [low, high], after applying all (1-1/p) for p <= sqrt(high),
        # phi[m] is correct UNLESS m has a prime factor q > sqrt(high).
        # If m has such a factor q, then m = q * r where r < sqrt(high) (since m <= high < q^2).
        # We already divided by all prime factors of r (since r < sqrt(high)), so the (1-1/q)
        # factor is missing.
        #
        # To detect: for each index i, let m = low + i. If m has a prime factor q > sqrt(high),
        # then m/q < sqrt(high) and m/q is an integer. We can check: does m have a divisor d < sqrt(high)
        # such that m/d > sqrt(high) and m/d is prime?
        #
        # Simpler approach: For each i, compute the "reduced" value by dividing out all small primes.
        # But do it efficiently using a separate array and only one pass per prime.
        
        # Efficient approach: track the largest prime factor > sqrt(high) for each position
        # For each position i, if (low+i) is divisible by some q > sqrt(high), then (low+i)/q < sqrt(high)
        # and (low+i)/q must be composite or 1 (since we've removed all small prime factors).
        # Actually (low+i)/q could be prime too, but then it would be <= sqrt(high).
        
        # Let's use a different strategy: for each position, check if the number is prime
        # or has a large prime factor by checking if it equals a product of a small number and a large prime.
        
        # Most efficient: for each small number r in [1, sqrt(high)], check if r * q falls in [low, high]
        # for some prime q > sqrt(high). This is equivalent to: for each r, find q such that
        # low <= r*q <= high and q > sqrt(high), q prime.
        # Then phi[r*q] should have (1-1/q) applied.
        
        # But this is complex. Let's use a simpler but still fast approach:
        # For each position i, compute rem = low+i. Divide rem by all primes <= sqrt(high).
        # If the result > 1, apply (1-1/result) to phi[i].
        # Do this with a single pass using a reduced array.
        
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
            # Divide by p as many times as needed
            for _ in range(30):  # enough for 64-bit numbers
                mask = reduced[indices] % p == 0
                if not np.any(mask):
                    break
                reduced[indices[mask]] = reduced[indices[mask]] // p
        
        # Now apply the remaining factor
        mask = reduced > 1
        if np.any(mask):
            phi[mask] = phi[mask] - phi[mask] // reduced[mask]
        
        return phi

    best_k = 1
    best_start = 1

    # Phase 1: Segmented sieve search using NumPy over large ranges
    # Focus on ranges where long runs are more likely: higher numbers have more prime factors
    # and thus more variation in phi values
    CHUNK_SIZE = 500000  # 500k numbers per chunk - faster to process
    
    # Search in strategically chosen ranges
    # Higher numbers tend to have longer runs of distinct totients
    search_ranges = [
        (10**11, 10**12),      # high range
        (10**10, 10**11),      # mid-high
        (10**9, 10**10),       # mid
        (10**8, 10**9),        # lower-mid
        (10**7, 10**8),        # lower
        (10**6, 10**7),        # low-mid
        (10**5, 10**6),        # low
        (1, 10**5),            # very low
    ]
    
    for range_start, range_end in search_ranges:
        if time_remaining() < 10.0:
            break
        search_start = range_start
        while search_start < range_end and time_remaining() > 5.0:
            chunk_end = min(search_start + CHUNK_SIZE - 1, range_end, 10**12)
            try:
                phi_vals = segmented_phi_fast(search_start, chunk_end)
                k, s = find_longest_distinct_run(phi_vals, search_start)
                if k > best_k:
                    best_k = k
                    best_start = s
                    if best_k >= 200:
                        break
            except MemoryError:
                CHUNK_SIZE = 100000
                continue
            except Exception:
                pass
            
            search_start = chunk_end + 1
        if best_k >= 200:
            break

    # Phase 2: Targeted search around primorial multiples
    random.seed(42)
    
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
    
    for prim in primorials:
        if time_remaining() < 5.0:
            break
        max_mult = min(10**12 // prim, 100000)
        for _ in range(min(50, max_mult)):
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
            for offset in [-10, -5, -1, 1, 5, 10]:
                s2 = start + offset
                if s2 < 1 or s2 + 200 > 10**12:
                    continue
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_start = s2

    # Phase 3: Random search in promising ranges
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(200):
            if time_remaining() < 2.0:
                break
            # Bias towards higher ranges
            r = random.random()
            if r < 0.3:
                s = random.randint(10**11, 10**12 - 200)
            elif r < 0.6:
                s = random.randint(10**10, 10**11 - 200)
            elif r < 0.8:
                s = random.randint(10**9, 10**10 - 200)
            else:
                s = random.randint(10**6, 10**9 - 200)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 4: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        for delta in range(-50, 51):
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
