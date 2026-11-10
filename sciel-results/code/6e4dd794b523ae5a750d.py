"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_fast(n):
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def find_run(s, max_k=200):
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best = (1, 1)

    # Phase 1: Small range search [1, 100000] for baseline
    end = min(100000, 10**12)
    vals = [0] + [phi_fast(x) for x in range(1, end + 1)]
    for s in range(1, end - 200):
        seen = set()
        k = 0
        while s + k <= end and vals[s + k] not in seen:
            seen.add(vals[s + k])
            k += 1
        if k > best[0]:
            best = (k, s)
        if time_left() < 30:
            break

    # Phase 2: Random sampling in [1, 10^12]
    random.seed(42)
    attempts = 0
    while time_left() > 5:
        s = random.randint(1, 10**12 - 200)
        k = find_run(s)
        if k > best[0]:
            best = (k, s)
        attempts += 1

    return {'start': best[1], 'length': best[0]}
# EVOLVE-BLOCK-END
