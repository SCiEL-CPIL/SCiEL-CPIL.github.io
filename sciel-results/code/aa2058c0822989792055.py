import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Precompute primes up to a larger limit for better coverage
    P_MAX = 500000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    
    # Helper to check if n is p-smooth using precomputed primes
    def is_p_smooth_fast(n, p, primes_list):
        if n <= p:
            return True
        rem = n
        for q in primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
        if rem == 1:
            return True
        if rem <= p:
            return True
        return False

    def find_max_k(p, max_k=200):
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth_fast(p2 + j, p, primes):
                break
            k += 1
        return k

    best_k = 1
    best_p = 2

    # Strategy: Try larger primes first since they have more available
    # prime factors and are more likely to support long p-smooth runs.
    # Process primes in descending order.
    for idx in range(len(primes) - 1, -1, -1):
        p = primes[idx]
        if time.time() - start_time > DEADLINE:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
