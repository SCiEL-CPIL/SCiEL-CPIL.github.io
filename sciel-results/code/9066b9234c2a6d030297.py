"""Rewritten program: fast primality, Pollard-Brent factorization, and search for
long runs of p^2+1 .. p^2+k where every factor is <= p.

Strategy:
- Precompute primes up to a modest limit (e.g., 10^6) for trial division.
- Use deterministic Miller-Rabin for 64-bit integers.
- Use Pollard-Brent for factoring composites.
- For each candidate p (prime), check k=1,2,...,200: factor p^2+i and verify
  all prime factors <= p. Stop at first i where a factor > p appears.
- Use a deadline guard to stop before 150s.
- Search over primes up to a reasonable upper bound, prioritizing larger p
  (larger p allows larger factors, so longer runs are more likely).
"""

import time
import math
import random
from functools import reduce

DEADLINE = 140.0  # safety margin before 150s

# ---------- Primality ----------

def is_prime(n):
    """Deterministic Miller-Rabin for n < 2^64."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 = d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard-Brent ----------

def pollard_brent(n):
    """Return a non-trivial factor of n using Pollard-Brent."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        g = r = q = 1
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
        if g != n:
            return g

# ---------- Factorization ----------

def factorize(n, primes_small):
    """Return sorted list of prime factors of n (with multiplicity)."""
    factors = []
    # Trial divide by small primes
    for p in primes_small:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        # n is either prime or product of two large primes
        if is_prime(n):
            factors.append(n)
        else:
            # Use Pollard-Brent
            stack = [n]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if is_prime(m):
                    factors.append(m)
                else:
                    d = pollard_brent(m)
                    stack.append(d)
                    stack.append(m // d)
    return factors

# ---------- Precompute small primes ----------

def sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i in range(limit + 1) if is_p[i]]

SMALL_PRIMES = sieve(100000)

# ---------- Core check ----------

def max_k_for_p(p, primes_small):
    """Return the maximum k such that all of p^2+1..p^2+k have all prime factors <= p."""
    k = 0
    for i in range(1, 201):
        n = p * p + i
        facs = factorize(n, primes_small)
        if any(f > p for f in facs):
            break
        k += 1
    return k

# ---------- Search ----------

def construct():
    start = time.time()
    best_k = 0
    best_p = 3

    # Search over primes. Start from larger primes since they allow larger factors.
    # We'll iterate over primes in a smart order: try primes up to some bound.
    # Given time constraints, we'll generate primes up to a limit and test them.
    
    # Generate primes up to 10^7 for search
    search_limit = 10_000_000
    primes = sieve(search_limit)
    
    # Test primes in descending order of likelihood of long runs:
    # Larger p tends to allow longer runs. But also, p where p^2+1 has small factors is good.
    # We'll test primes from larger to smaller, but with a time guard.
    
    # Actually, let's test a mix: first a few large primes, then sweep.
    # For efficiency, test primes in order but skip if time is running out.
    
    for p in primes:
        if time.time() - start > DEADLINE - 5:
            break
        if p < 3:
            continue
        k = max_k_for_p(p, SMALL_PRIMES)
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good k, we can try to extend search a bit
            if best_k >= 10:
                # Continue searching but be more careful
                pass
    
    # If we still have time, try some larger primes by testing specific candidates
    remaining = DEADLINE - (time.time() - start)
    if remaining > 10:
        # Try primes in [10^7, 10^8] range by sampling
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() - start > DEADLINE - 3:
                break
            # Generate a random odd number in range
            p = random.randint(10_000_001, 100_000_000)
            if p % 2 == 0:
                p += 1
            if is_prime(p):
                k = max_k_for_p(p, SMALL_PRIMES)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
