"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to sqrt(10^12) = 10^6 using sieve
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        primes = np.where(is_prime)[0]
        return primes

    # Build prime list up to 10^6
    MAX_PRIME = 10**6
    primes = sieve_primes(MAX_PRIME)
    primes_list = primes.tolist()

    def phi_fast(n):
        """Compute Euler's totient function using precomputed primes."""
        if n <= 1:
            return 0
        result = n
        y = n
        for p in primes_list:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_best_in_range(start, length, max_check=200):
        """Find the longest run of distinct totients starting from 'start' within 'length' numbers."""
        if time_left() < 5.0:
            return (0, 0)
        seen = set()
        k = 0
        for i in range(length):
            if time_left() < 2.0:
                break
            n = start + i
            if n > 10**12:
                break
            v = phi_fast(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
            if k >= max_check:
                break
        return (k, start)

    best_k = 1
    best_start = 1

    # Strategy: Search in random ranges up to 10^12
    # Longer runs are more likely in larger ranges
    random.seed(42)

    # First, do a quick local search near 1 to establish a baseline
    for s in range(1, 1000):
        if time_left() < 10.0:
            break
        k, st = find_best_in_range(s, 200)
        if k > best_k:
            best_k = k
            best_start = st

    # Now search in random ranges up to 10^12
    # Use larger starting points where distinct totients are more common
    while time_left() > 15.0:
        # Pick random start points in various ranges
        # Larger numbers tend to have more distinct totients
        if random.random() < 0.3:
            # Small range
            s = random.randint(1, 10**6)
        elif random.random() < 0.6:
            # Medium range
            s = random.randint(10**6, 10**9)
        else:
            # Large range
            s = random.randint(10**9, 10**12 - 200)

        # Ensure we don't exceed 10^12
        s = min(s, 10**12 - 200)
        s = max(s, 1)

        k, st = find_best_in_range(s, 200)
        if k > best_k:
            best_k = k
            best_start = st

        # Also try nearby starts if we found something good
        if k >= 10:
            for offset in [-50, -25, -10, 10, 25, 50]:
                if time_left() < 5.0:
                    break
                ns = s + offset
                if ns >= 1 and ns + 200 <= 10**12:
                    k2, st2 = find_best_in_range(ns, 200)
                    if k2 > best_k:
                        best_k = k2
                        best_start = st2

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
