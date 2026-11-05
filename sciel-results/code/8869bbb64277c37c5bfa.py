import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    # Precompute primes up to a reasonable limit.
    # We need primes up to P_MAX to use for trial division.
    # If we search p up to 100,000, we need primes up to 100,000.
    P_MAX = 100000
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
        # We only need to divide by primes q <= p.
        # Optimization: if q*q > rem, rem is prime.
        # If rem > p, it's not p-smooth.
        # If rem <= p, it is p-smooth (since rem is prime and <= p).
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

    # Test primes in increasing order
    # Most primes will fail early, so this is fast.
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a very high k, we might want to continue searching nearby,
            # but for now, just keep going.

    return {'p': best_p, 'k': best_k}
