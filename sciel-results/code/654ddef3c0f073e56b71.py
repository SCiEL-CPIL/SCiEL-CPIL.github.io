import time
import math
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s limit
    best_k = 1
    best_s = 1

    # Sieve of Eratosthenes up to 10^6 for fast factorization
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def phi(x):
        """Compute Euler's totient function using precomputed primes."""
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

    # Strategy: search for long runs in regions where numbers are "generic"
    # (not too many small prime factors), which tends to give distinct totients.
    # We'll scan multiple candidate regions and track the best run found.

    # First, do a quick scan of small numbers to establish a baseline
    max_scan = 50000
    phi_cache = [0] * (max_scan + 1)
    phi_cache[1] = 1
    for i in range(2, max_scan + 1):
        if time.time() > deadline - 10:
            break
        phi_cache[i] = phi(i)

    # Find best run in the scanned region
    for s in range(1, max_scan - 100):
        if time.time() > deadline - 5:
            break
        seen = set()
        k = 0
        while s + k <= max_scan and phi_cache[s + k] not in seen:
            seen.add(phi_cache[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    # Now search in larger regions. We'll pick random starting points
    # and scan forward to find runs.
    # Focus on numbers that are products of large primes or have few small factors.

    # Try several random regions across [1, 10^12]
    random.seed(42)
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break

        # Choose a random start point
        # Bias towards numbers that are "smooth" in a good way
        start = random.randint(1, 10**12 - 200)

        # Scan forward from start
        seen = set()
        k = 0
        for offset in range(200):
            n = start + offset
            if n > 10**12:
                break
            if time.time() > deadline - 2:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1

        if k > best_k:
            best_k = k
            best_s = start

    # Also try some structured searches:
    # Numbers near primes often have phi values that are distinct.
    # Try regions around large primes.

    # Additional targeted search: look for runs starting at numbers
    # that are themselves prime or have large prime factors.
    for trial in range(100):
        if time.time() > deadline - 5:
            break

        # Pick a random number and check if it's in a "good" region
        base = random.randint(10**8, 10**12 - 200)
        # Round to nearest multiple of a large number to avoid obvious patterns
        start = base

        seen = set()
        k = 0
        for offset in range(200):
            n = start + offset
            if n > 10**12:
                break
            if time.time() > deadline - 2:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1

        if k > best_k:
            best_k = k
            best_s = start

    return {'start': best_s, 'length': best_k}
