"""Rewritten program: find long runs of consecutive integers with distinct totients."""
import time
import math
import random
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def sieve_primes(limit):
        """Return list of primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # Precompute primes up to 10^7 for fast factorization of larger numbers
    primes = sieve_primes(10000000)

    def phi_factorize(n, primes):
        """Compute phi(n) using trial division by precomputed primes."""
        if n == 1:
            return 1
        result = n
        y = n
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

    def phi_batch(start, length, primes):
        """Compute phi for range [start, start+length)."""
        vals = [0] * length
        for i in range(length):
            vals[i] = phi_factorize(start + i, primes)
        return vals

    def find_best_run_sliding(start, end, primes):
        """Find best run using sliding window with phi computation.
        
        Uses a proper two-pointer sliding window to find the longest
        contiguous subsequence where all phi values are distinct.
        """
        if start >= end:
            return 1, start
        
        best_k = 1
        best_s = start
        
        # We'll compute phi values on the fly and use a sliding window
        # left = left pointer of current window
        # right = right pointer (current position)
        # freq = dictionary mapping phi value to count in current window
        
        left = start
        freq = {}
        k = 0
        
        for right in range(start, end):
            if time_left() < 5.0:
                break
            
            v = phi_factorize(right, primes)
            
            # Add v to window
            if v in freq:
                freq[v] += 1
            else:
                freq[v] = 1
            
            # Shrink window from left until no duplicates
            while freq[v] > 1:
                lv = phi_factorize(left, primes)
                freq[lv] -= 1
                if freq[lv] == 0:
                    del freq[lv]
                left += 1
            
            # Update best
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = left
        
        return best_k, best_s

    # Strategy:
    # 1. Search in several strategic ranges
    # 2. Use larger ranges for better chances of finding long runs
    # 3. Prioritize ranges where numbers have varied factorizations

    best_k = 1
    best_s = 1

    # Phase 1: Search small ranges quickly to establish baseline
    if time_left() > 30:
        k, s = find_best_run_sliding(1, 50000, primes)
        if k > best_k:
            best_k = k
            best_s = s

    # Phase 2: Search in larger strategic ranges
    search_ranges = []
    
    # Strategic ranges: near powers of 10 and random locations
    for base in [10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
        search_ranges.append((base, base + 100000))
    
    # Random ranges
    random.seed(42)
    for _ in range(30):
        lo = random.randint(10**5, 10**11)
        search_ranges.append((lo, lo + 100000))

    for lo, hi in search_ranges:
        if time_left() < 10:
            break
        if hi > 10**12:
            hi = 10**12
        if lo >= hi:
            continue
        k, s = find_best_run_sliding(lo, hi, primes)
        if k > best_k:
            best_k = k
            best_s = s
            # If we found a good run, try to extend it in both directions
            if best_k >= 5 and time_left() > 20:
                ext_lo = max(1, best_s - 50000)
                ext_hi = min(10**12, best_s + best_k + 50000)
                k2, s2 = find_best_run_sliding(ext_lo, ext_hi, primes)
                if k2 > best_k:
                    best_k = k2
                    best_s = s2

    # Phase 3: If we have time, do more extensive random search
    if time_left() > 30:
        for _ in range(50):
            if time_left() < 10:
                break
            lo = random.randint(10**6, 10**11)
            hi = lo + 200000
            if hi > 10**12:
                hi = 10**12
            k, s = find_best_run_sliding(lo, hi, primes)
            if k > best_k:
                best_k = k
                best_s = s

    # Ensure constraints
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200

    return {'start': best_s, 'length': best_k}
