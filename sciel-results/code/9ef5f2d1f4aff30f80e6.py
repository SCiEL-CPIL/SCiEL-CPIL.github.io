"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def construct():
    deadline = time.time() + 140  # safety margin before 150s
    
    best_k = 0
    best_p = 0
    
    # Precompute small primes for trial division speedup
    # Use a sieve to generate primes up to 200000
    def primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    small_primes = primes_up_to(200000)
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p, using trial division first."""
        if n <= 1:
            return True
        # Trial division by small primes
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > n:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
        if n == 1:
            return True
        # n is now either prime or has no small prime factors <= min(p, sqrt(original_n))
        if is_prime(n):
            return n <= p
        # n is composite with no small factors; use Pollard's rho
        facs = factor(n)
        return all(f <= p for f in facs)
    
    # Generate primes up to 100000 for the main search
    primes_list = primes_up_to(100000)
    
    # First pass: check all primes up to 100000
    for p in primes_list:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Second pass: try larger primes in sampled ranges
    if time.time() < deadline - 5:
        for start in [100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000]:
            if time.time() > deadline - 5:
                break
            end = min(start * 2, 1000000000)
            if end <= start:
                continue
            # Use segmented sieve for a window
            window_size = min(200000, end - start)
            if window_size <= 0:
                continue
            sieve_start = start
            sieve_end = start + window_size
            sieve = [True] * (window_size)
            for sp in small_primes:
                if sp * sp > sieve_end:
                    break
                first = ((sieve_start + sp - 1) // sp) * sp
                for j in range(first - sieve_start, window_size, sp):
                    sieve[j] = False
            # Check primality of candidates
            for i in range(window_size):
                if time.time() > deadline - 5:
                    break
                cand = sieve_start + i
                if cand < 2:
                    continue
                if sieve[i] and is_prime(cand):
                    # Check this prime
                    k = 0
                    for ii in range(1, 201):
                        n = cand * cand + ii
                        if is_p_smooth(n, cand):
                            k += 1
                        else:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = cand
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
