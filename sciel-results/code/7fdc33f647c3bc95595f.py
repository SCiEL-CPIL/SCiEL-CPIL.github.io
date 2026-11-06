import time
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0

    _MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

    def is_prime(n):
        if n < 2:
            return False
        for p in _MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in _MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    SIEVE_LIMIT = 5000000
    primes = sieve_primes(SIEVE_LIMIT)
    
    # Small primes for quick division (primes < 10000)
    small_primes = [q for q in primes if q < 10000]

    def is_p_smooth_fast(n, p):
        """Check if n is p-smooth using trial division with small primes first."""
        if n <= 1:
            return True
        m = n
        # Divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        # m is now either prime or has a factor > 10000
        # Check if m is prime
        if is_prime(m):
            return m <= p
        # m is composite with factors > 10000
        # Use trial division with larger primes
        for q in primes:
            if q < 10000:
                continue
            if q > p:
                break
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        if is_prime(m):
            return m <= p
        # m is composite with all factors > p (since we've tried all primes <= p)
        return False

    def check_run(p):
        k = 0
        p_sq = p * p
        for i in range(1, 201):
            if time.time() > DEADLINE - 2:
                break
            if is_p_smooth_fast(p_sq + i, p):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Check primes in a strategic order: start with medium-sized primes
    # where p^2 is large enough to have many small factors, but p is still
    # manageable for trial division
    # Try primes from ~10000 up to SIEVE_LIMIT, then smaller ones if time permits
    n_primes = len(primes)
    
    # First pass: check primes in range [10000, SIEVE_LIMIT]
    # These are more likely to have long runs because p^2 is large
    idx_start = 0
    for idx, q in enumerate(primes):
        if q >= 10000:
            idx_start = idx
            break
    
    for idx in range(idx_start, n_primes):
        p = primes[idx]
        if time.time() > DEADLINE - 5:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # If time remains, check smaller primes
    if time.time() < DEADLINE - 10:
        for idx in range(0, idx_start):
            p = primes[idx]
            if time.time() > DEADLINE - 5:
                break
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
