import math
import time
import random

def construct():
    deadline = time.time() + 115.0

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Sieve primes up to 100000 for trial division
    SIEVE_LIMIT = 100000
    primes = sieve(SIEVE_LIMIT)
    small_primes = [p for p in primes if p <= 1000]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes_mr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes_mr:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def pollard_brent(n):
        """Pollard's Brent cycle detection factorization."""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def is_p_smooth(n, p):
        """Check if n is p-smooth (all prime factors <= p)."""
        if n <= 1:
            return True
        # First, trial divide by small primes
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
        if n <= 1:
            return True
        if n <= p:
            return True
        # n > p: try to factor it further
        # If n is prime, it's a factor > p -> fail
        if is_prime_mr(n):
            return False
        # n is composite; try Pollard-Brent
        try:
            factor = pollard_brent(n)
            if factor > p:
                return False
            n //= factor
            return is_p_smooth(n, p)
        except:
            return False

    def check_p(p):
        """Check how many consecutive numbers starting from p^2+1 are p-smooth."""
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            if time.time() > deadline:
                break
            v = p_sq + k_val
            if is_p_smooth(v, p):
                k = k_val
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Check p = 2
    k = check_p(2)
    if k > best_k:
        best_k = k
        best_p = 2

    # Check primes in descending order (larger primes first)
    # This is more efficient because larger p allows more small prime factors
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p > 10**9:
            continue
        if p < 2:
            continue

        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Try some larger primes using Miller-Rabin
    random.seed(42)
    checked = set(primes)
    for _ in range(5000):
        if time.time() > deadline:
            break
        candidate = random.randint(100000, 1000000)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not is_prime_mr(candidate):
            continue
        p = candidate
        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
