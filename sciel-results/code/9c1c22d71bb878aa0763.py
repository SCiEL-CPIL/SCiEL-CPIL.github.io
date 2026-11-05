import math
import time
import random

def construct():
    deadline = time.time() + 113.0

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    SIEVE_LIMIT = 5000000
    primes = sieve(SIEVE_LIMIT)

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for sp in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def check_run_segmented(p, max_k=200):
        """Segmented sieve with early termination per number."""
        p_sq = p * p
        lo = p_sq + 1
        hi = p_sq + max_k
        size = max_k
        rem = [lo + i for i in range(size)]
        # Track which numbers are still "alive" (not yet determined to be non-smooth)
        alive = [True] * size
        
        for q in primes:
            if q > p:
                break
            start = ((lo + q - 1) // q) * q
            if start < lo:
                start += q
            for j in range(start, hi + 1, q):
                idx = j - lo
                if not alive[idx]:
                    continue
                while rem[idx] % q == 0:
                    rem[idx] //= q
                # Early check: if rem[idx] > p and rem[idx] is not divisible by any future prime, it's non-smooth
                # But we can't know that yet, so just continue
            # Optimization: if q > sqrt(hi), no more composites to sieve
            if q * q > hi:
                break
        
        k = 0
        for i in range(size):
            if rem[i] <= p:
                k = i + 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Check small primes quickly
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        k = check_run_segmented(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Search in DESCENDING order: larger primes have higher p-smoothness probability
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p < 97:
            break
        k = check_run_segmented(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 180:
            break

    # Try random primes beyond sieve range
    random.seed(42)
    checked = set(primes)
    for _ in range(3000):
        if time.time() > deadline:
            break
        c = random.randint(5000000, 10**7)
        if c in checked:
            continue
        checked.add(c)
        if not is_prime_mr(c):
            continue
        k = check_run_segmented(c)
        if k > best_k:
            best_k = k
            best_p = c
        if best_k >= 180:
            break

    return {'p': best_p, 'k': best_k}
