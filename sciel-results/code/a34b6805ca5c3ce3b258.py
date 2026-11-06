"""Improved program: targeted search for long runs of distinct totients."""
import time
import math
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes for trial division
    def sieve_primes(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    small_primes = sieve_primes(10000)  # primes up to 10000 for quick factorization

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

    def compute_phi_window(start, length, primes=None):
        """Compute phi for a window of consecutive integers."""
        if primes is None:
            primes = small_primes
        vals = []
        for i in range(length):
            vals.append(phi_trial(start + i, primes))
        return vals

    def find_longest_distinct_run(phi_vals):
        """Find the longest run of consecutive distinct values in phi_vals.
        Uses a sliding window approach: when we see a duplicate, we need to
        find where it first appeared in the current window and move the start
        past that position.
        """
        if len(phi_vals) == 0:
            return 0, 0
        
        best_k = 1
        best_start = 0
        # For each position, track the last seen index of each value
        last_seen = {}
        window_start = 0  # start of current valid window
        
        for i, v in enumerate(phi_vals):
            if v in last_seen and last_seen[v] >= window_start:
                # Duplicate found within current window
                # Move window start past the previous occurrence
                window_start = last_seen[v] + 1
            last_seen[v] = i
            
            current_len = i - window_start + 1
            if current_len > best_k:
                best_k = current_len
                best_start = window_start
        
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

    # Strategy: Search in regions where numbers are "smooth" or have many prime factors
    # Long runs of distinct phi values are more likely when numbers have varied factorizations.
    
    best_k = 1
    best_start = 1

    # Phase 1: Search small ranges with NumPy-accelerated sieve for speed
    def compute_phi_sieve_numpy(limit):
        """Compute phi for all numbers up to limit using NumPy vectorization."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                # Vectorized update for all multiples of i
                multiples = np.arange(i, limit + 1, i)
                phi[multiples] -= phi[multiples] // i
        return phi

    # Try sieve up to 10^6 first
    sieve_limit = 10**6
    if time_remaining() > 20.0:
        try:
            phi_vals = compute_phi_sieve_numpy(sieve_limit)
            k, s = find_longest_distinct_run(phi_vals[1:])  # skip index 0
            if k > best_k:
                best_k = k
                best_start = s + 1
        except (MemoryError, Exception):
            pass

    # Try larger sieve if time permits
    if time_remaining() > 15.0:
        sieve_limit = 10**7
        try:
            phi_vals = compute_phi_sieve_numpy(sieve_limit)
            k, s = find_longest_distinct_run(phi_vals[1:])
            if k > best_k:
                best_k = k
                best_start = s + 1
        except (MemoryError, Exception):
            pass

    # Try even larger sieve if time permits
    if time_remaining() > 10.0:
        sieve_limit = 2 * 10**7
        try:
            phi_vals = compute_phi_sieve_numpy(sieve_limit)
            k, s = find_longest_distinct_run(phi_vals[1:])
            if k > best_k:
                best_k = k
                best_start = s + 1
        except (MemoryError, Exception):
            pass

    # Phase 2: Targeted search in larger ranges
    # Regions where numbers are products of many small primes tend to have distinct phi values
    # because phi(n) = n * prod(1 - 1/p) varies significantly.
    
    # Generate candidate starts based on products of small primes
    # and search around them.
    
    random.seed(42)
    
    # Try windows around multiples of primorials
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]
    
    for prim in primorials:
        if time_remaining() < 5.0:
            break
        # Try a few multiples
        max_mult = min(10**12 // prim, 10000)
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
            # Also try start - 1, start + 1, etc.
            for offset in [-10, -5, -1, 1, 5, 10]:
                s2 = start + offset
                if s2 < 1 or s2 + 200 > 10**12:
                    continue
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_start = s2

    # Phase 3: Random search in mid-range
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(200):
            if time_remaining() < 2.0:
                break
            s = random.randint(10**6, 10**11)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 4: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        # Try expanding around the best start
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
