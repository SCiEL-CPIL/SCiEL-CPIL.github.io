"""Rewritten program: faster primality/factorization, smarter p-search, deadline guard."""
import time
import random
import math

# Deterministic Miller-Rabin bases for n < 2^64
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    """Deterministic Miller-Rabin primality test for n < 2^64."""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 21, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
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

def _pollard_brent(n):
    """Pollard's rho with Brent's improvement. Returns a non-trivial factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = random.randrange(1, 100)
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
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def _factor(n, factor_list=None):
    """Recursively factor n using trial division for small primes, then Pollard-Brent."""
    if factor_list is None:
        factor_list = []
    if n <= 1:
        return factor_list
    if n < 1000000:
        # Trial division for small numbers
        d = 2
        while d * d <= n:
            while n % d == 0:
                factor_list.append(d)
                n //= d
            d += 1
        if n > 1:
            factor_list.append(n)
        return factor_list
    # Try small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        while n % p == 0:
            factor_list.append(p)
            n //= p
    if n == 1:
        return factor_list
    if _is_prime_mr(n):
        factor_list.append(n)
        return factor_list
    # Use Pollard-Brent
    d = _pollard_brent(n)
    _factor(d, factor_list)
    _factor(n // d, factor_list)
    return factor_list

def _max_prime_factor(n):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    factors = _factor(n)
    return max(factors)

def _sieve_primes(limit):
    """Sieve of Eratosthenes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Pre-sieve primes up to a reasonable limit for fast p-generation
    # We'll search p up to maybe 200000 initially, then extend if time permits
    sieve_limit = 200000
    primes = _sieve_primes(sieve_limit)
    
    best_k = 0
    best_p = 2
    
    # For each prime p, check how many consecutive integers p^2+1, ..., p^2+k
    # have all prime factors <= p
    for p in primes:
        if time.time() > deadline - 5:
            break
        if p > 10**9:
            break
        p_sq = p * p
        k = 0
        # Check consecutive integers
        # We can short-circuit: if any prime factor > p, stop
        while k < 200:
            n = p_sq + k + 1
            # Quick check: if n is prime and n > p, fail
            # But n could be composite with all factors <= p
            # Need to factor n
            # Optimization: first check divisibility by small primes up to p
            # Actually we need to verify all prime factors <= p
            # If n has a prime factor > p, then that factor is a prime factor of n
            # that exceeds p. So we need to find if any prime factor of n exceeds p.
            # Equivalent: after removing all prime factors <= p, the remainder must be 1.
            
            # Trial divide by primes up to min(p, sqrt(n))
            # But p can be large. Let's use a hybrid approach.
            # First, check if n is prime. If n <= p, it's fine. If n > p and prime, fail.
            # If composite, factor it.
            
            # Quick path: if n <= p, all factors <= p automatically
            if n <= p:
                k += 1
                continue
            
            # n > p: need to verify no prime factor > p
            # Factor n
            temp = n
            ok = True
            # Divide by primes up to min(p, sqrt(temp))
            # Use our primes list if p <= sieve_limit, else trial divide
            if p <= sieve_limit:
                # Use precomputed primes
                for q in primes:
                    if q > p:
                        break
                    if q * q > temp:
                        break
                    while temp % q == 0:
                        temp //= q
                if temp > 1:
                    # temp is either prime or has a factor > p
                    if temp > p:
                        ok = False
            else:
                # p > sieve_limit, need to trial divide up to min(p, sqrt(temp))
                # This is expensive. Use Pollard-Brent to factor and check.
                factors = _factor(n)
                if any(f > p for f in factors):
                    ok = False
            
            if not ok:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # Print progress
            # print(f"p={p}, k={k}")
    
    # If we have time, try larger primes
    if time.time() < deadline - 10:
        # Extend sieve or use Miller-Rabin for larger primes
        # Try primes in ranges beyond sieve_limit
        # Generate candidates and test primality
        search_up_to = min(500000, 10**9)
        if sieve_limit < search_up_to:
            # Continue from sieve_limit + 1
            candidate = sieve_limit + 1
            if candidate % 2 == 0:
                candidate += 1
            while candidate <= search_up_to:
                if time.time() > deadline - 5:
                    break
                if _is_prime_mr(candidate):
                    p = candidate
                    if p > 10**9:
                        break
                    p_sq = p * p
                    k = 0
                    while k < 200:
                        n = p_sq + k + 1
                        if n <= p:
                            k += 1
                            continue
                        factors = _factor(n)
                        if any(f > p for f in factors):
                            break
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_p = p
                candidate += 2
    
    return {'p': best_p, 'k': best_k}
