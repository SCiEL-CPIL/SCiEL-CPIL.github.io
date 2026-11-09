"""Rewritten program with fast factorization and deadline-aware search."""
import time
import random
from math import isqrt, gcd

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # ---------- Pollard-Brent ----------
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n - 1)
            c = random.randrange(1, n - 1)
            m = 128
            g = r = q = 1
            x = ys = y
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    # ---------- Factorization ----------
    def factor(n):
        """Return list of prime factors (with multiplicity) of n."""
        if n <= 1:
            return []
        if is_prime(n):
            return [n]
        # Try small primes first
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return [p] + factor(n // p)
        d = pollard_brent(n)
        return factor(d) + factor(n // d)

    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n <= 1:
            return 1
        factors = factor(n)
        return max(factors)

    # ---------- Check if all numbers p^2+1 .. p^2+k have LPF <= p ----------
    def check_run(p, max_k=200):
        """Return the maximal k such that p^2+1..p^2+k all have LPF <= p."""
        k = 0
        base = p * p
        while k < max_k:
            if time.time() > DEADLINE:
                break
            n = base + k + 1
            lpf = largest_prime_factor(n)
            if lpf > p:
                break
            k += 1
        return k

    # ---------- Sieve primes up to 10^6 for initial search ----------
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(SIEVE_LIMIT) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]

    # ---------- Search ----------
    best_k = 0
    best_p = 2

    # Phase 1: Deterministic search over primes up to 10^6
    for p in primes:
        if time.time() > DEADLINE:
            break
        if p > 10**6:
            break
        # Quick pre-filter: check a few values early to skip bad candidates
        # p^2+1 must have all factors <= p. If p^2+1 is prime and > p, skip.
        # Actually p^2+1 > p always for p>=2, so if p^2+1 is prime it's > p -> fail.
        # But p^2+1 could be composite with small factors.
        k = check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search for larger primes if time remains
    remaining = DEADLINE - time.time()
    if remaining > 5:
        # Try random primes up to 10^9
        # Generate random odd numbers and test primality
        attempts = 0
        max_attempts = 500
        while time.time() < DEADLINE - 2 and attempts < max_attempts:
            attempts += 1
            # Random number in [10^6, 10^9)
            n = random.randint(10**6, 10**9 - 1)
            if n % 2 == 0:
                n += 1
            if not is_prime(n):
                continue
            k = check_run(n, 200)
            if k > best_k:
                best_k = k
                best_p = n

    return {'p': best_p, 'k': best_k}
