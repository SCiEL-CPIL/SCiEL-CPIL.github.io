import math
import time
import random

DEADLINE = time.time() + 115.0  # Safety margin: evaluator kills at 120s, we stop at 115s

def is_prime(n):
    """Deterministic Miller-Rabin for n < 3,474,749,660,383"""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Bases sufficient for n < 3,474,749,660,383
    witnesses = [2, 3, 5, 7, 11, 13, 17]
    for a in witnesses:
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

def pollard_brent(n):
    """Pollard's Rho algorithm with Brent's cycle detection."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, q, r, x, ys = 1, 1, 1, 0, y
        while g == 1:
            x = y
            for i in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for i in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r += m
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def factorize(n, factors):
    """Recursively factorize n into a list of prime factors."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    
    d = pollard_brent(n)
    factorize(d, factors)
    factorize(n // d, factors)

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n == 1:
        return 1
    factors = []
    factorize(n, factors)
    return max(factors) if factors else 1

def construct():
    best_k = 0
    best_p = 2
    start_time = time.time()
    
    # Strategy:
    # 1. Pre-compute small primes for a fast sieve.
    # 2. Iterate through candidate primes p.
    # 3. For each p, perform a fast sieve check: for i in 1..200, check if p^2+i has any small prime factor q < p.
    #    If it does, it's not p-smooth (unless the factor is p itself, but p^2+i is not divisible by p).
    #    Actually, if p^2+i is divisible by a prime q < p, then p^2+i is NOT p-smooth because q < p is allowed?
    #    Wait. The condition is "all prime factors at most p".
    #    So if p^2+i has a factor q <= p, that's fine.
    #    The only way p^2+i is NOT p-smooth is if it has a prime factor q > p.
    #    
    #    My previous logic in the seed was: "If val is prime and val > p, it fails."
    #    "If val is composite, largest factor must be <= p."
    #    
    #    How to sieve?
    #    If p^2 + i is divisible by a prime q where q > p, then it fails.
    #    We can't easily sieve for "has a factor > p".
    #    However, we CAN sieve for "has a factor < p".
    #    If p^2 + i has NO prime factors < p, then its smallest prime factor is >= p.
    #    If p^2 + i is composite, and its smallest prime factor is >= p, then all its prime factors are >= p.
    #    If p^2 + i is prime, then its only prime factor is itself. If p^2 + i > p, it fails.
    #    
    #    So, if we find that p^2 + i has NO prime factors < p, we must check if it is prime (and > p) or composite.
    #    If it is composite, its factors are all >= p. But we need them to be <= p.
    #    So if it is composite and has no factors < p, it must be a product of primes >= p.
    #    Since p^2 + i < p^2 + 201, for large p, it can only have at most 2 prime factors if they are large?
    #    Actually, if p is large, p^2 + i is roughly p^2.
    #    If it has no factors < p, it is either:
    #    1. Prime (fails if > p)
    #    2. Product of two primes both >= p.
    #       If it's a product of two primes q1, q2 >= p, then q1*q2 >= p^2.
    #       Since q1*q2 = p^2 + i, and i < 200, this is possible.
    #       In this case, the factors are q1, q2. Are they <= p?
    #       If q1 >= p and q2 >= p, then for them to be <= p, they MUST be equal to p.
    #       But p^2 + i is not divisible by p.
    #       So if p^2 + i has no prime factors < p, it CANNOT be p-smooth (unless it is 1, which it isn't).
    #       Wait.
    #       If p^2 + i is p-smooth, all its prime factors are <= p.
    #       Therefore, it MUST have at least one prime factor <= p.
    #       Since it is not divisible by p (p^2+i mod p = i mod p != 0 for i < p), it MUST have a prime factor q < p.
    #    
    #    CONCLUSION:
    #    A necessary condition for p^2 + i to be p-smooth is that it has at least one prime factor q < p.
    #    Therefore, if p^2 + i has NO prime factors < p, it is NOT p-smooth.
    #    
    #    This allows us to use a sieve!
    #    For a given p, we check i=1..200.
    #    For each i, we check if p^2 + i has any prime factor q < p.
    #    We can precompute primes up to some limit.
    #    But p can be up to 10^9. We can't precompute primes up to 10^9.
    #    However, if p is large, the probability that p^2 + i has a small factor is high?
    #    Actually, for any fixed i, p^2 + i is divisible by q if p^2 = -i mod q.
    #    This has a solution for p iff (-i/q) = 1.
    #    For a random p, the probability that p^2 + i is NOT divisible by any q < B is roughly 1/log(B) or similar?
    #    Actually, the density of numbers with no small factors is 1/log(N).
    #    So most numbers WILL have a small factor.
    #    
    #    So the sieve check is:
    #    For i in 1..200:
    #       n = p^2 + i
    #       Check if n has any prime factor q < p.
    #       If not, break (fail).
    #    
    #    How to check "has any prime factor q < p" fast?
    #    We can trial divide by primes up to some limit L.
    #    If we find a factor q < p, we are good (for this i).
    #    If we don't find a factor up to L, and L < p, we don't know for sure.
    #    But if L is large enough (e.g., 10^5), it's very likely we'll find a factor if one exists.
    #    If we don't find one, we can assume it might fail and do a full factorization?
    #    Or we can just say "likely fail" and skip?
    #    No, we need exactness.
    #    
    #    However, if p is large (e.g., 10^6), and we check primes up to 10^4, and find no factor,
    #    it's highly probable that the smallest factor is > 10^4.
    #    If the smallest factor is > 10^4, and p is 10^6, it could still be < p.
    #    So we can't skip.
    #    
    #    BUT, we can optimize the order.
    #    Check small primes first.
    #    If p^2 + i has a small factor (e.g., < 100), we pass the sieve for this i quickly.
    #    If it doesn't, we do a full factorization.
    #    This is much faster than doing full factorization for every i.
    
    # Generate small primes for quick trial division
    small_primes = []
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    for i in range(2, limit + 1):
        if sieve[i]:
            small_primes.append(i)
    
    p = 3
    while time.time() < DEADLINE:
        if is_prime(p):
            # Check run length
            k = 0
            while k < 200:
                val = p * p + k + 1
                # Quick check: does val have a prime factor < p?
                # If not, it's not p-smooth.
                # We check small primes first.
                has_small_factor = False
                for q in small_primes:
                    if q >= p:
                        break
                    if val % q == 0:
                        has_small_factor = True
                        break
                
                if not has_small_factor:
                    # No small factor found.
                    # It might still have a factor between limit and p.
                    # Or it might be prime (and > p).
                    # We must do full factorization to be sure.
                    # But this is rare.
                    lpf = largest_prime_factor(val)
                    if lpf > p:
                        break
                    # If lpf <= p, then it is p-smooth.
                    # Note: if lpf <= p, it must have a factor < p (since val is not div by p).
                    # So the "no small factor" case only happens if the smallest factor is > limit.
                    # In that case, full factorization confirms it.
                    
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break
        
        p += 2
        
    return {'p': best_p, 'k': best_k}
