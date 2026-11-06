"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

DEADLINE = 110.0  # safety margin below 120s evaluator limit

def _sieve_primes(limit):
    """Return list of all primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _miller_rabin(n, a_list):
    """Deterministic Miller-Rabin for n < 3.3e24 using given witnesses."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in a_list:
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

def _is_prime(n):
    """Fast primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic for n < 3,317,044,064,679,887,385,961,981
    return _miller_rabin(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])

def _pollard_brent(n, rand_func=None):
    """Pollard's rho with Brent's improvement."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if rand_func is None:
        rand_func = random.randrange
    y = rand_func(1, n - 1)
    c = rand_func(1, n - 1)
    m = 128
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
    if g == n:
        return None
    return g

def _largest_prime_factor(n, small_primes, p_limit):
    """
    Return the largest prime factor of n, or None if it exceeds p_limit.
    Uses trial division by small_primes first, then Pollard-Brent for remainder.
    Returns the largest prime factor found, or a value > p_limit if any factor exceeds it.
    """
    if n <= 1:
        return 1
    result = 1
    # Trial division by small primes
    for p in small_primes:
        if p > p_limit:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
            result = max(result, p)
    if n == 1:
        return result
    # Check if remaining n is prime
    if _is_prime(n):
        if n > p_limit:
            return n  # signal: exceeds limit
        return max(result, n)
    # Factor n using Pollard-Brent
    factors = []
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            factors.append(m)
            continue
        # Try to split
        d = None
        for attempt in range(50):
            d = _pollard_brent(m)
            if d is not None and 1 < d < m:
                break
        if d is None or d == m:
            # Fallback: m is likely prime
            factors.append(m)
        else:
            stack.append(d)
            stack.append(m // d)
    max_pf = max(factors)
    if max_pf > p_limit:
        return max_pf
    return max(result, max_pf)

def _check_run(p, small_primes, max_k=200, deadline=None):
    """
    Check how many consecutive integers p^2+1, ..., p^2+k have all prime factors <= p.
    Returns the maximum k found (1 to max_k, or 0 if p^2+1 already fails).
    """
    k = 0
    for i in range(1, max_k + 1):
        if deadline is not None and time.time() > deadline:
            break
        val = p * p + i
        lp = _largest_prime_factor(val, small_primes, p)
        if lp > p:
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Precompute small primes for trial division
    small_primes = _sieve_primes(1_000_000)
    
    best_k = 0
    best_p = 2
    
    # Strategy: search primes in increasing order of "potential" for long runs.
    # Larger primes tend to have longer runs because p^2 + i is less likely
    # to have a prime factor > p (the factors of p^2+i are spread out).
    
    # Generate primes up to a reasonable bound using sieve.
    sieve_limit = 2_000_000
    primes_list = _sieve_primes(sieve_limit)
    
    # Try primes from the end (largest) to the beginning
    n_primes = len(primes_list)
    candidates = list(reversed(primes_list))
    
    checked = 0
    for idx in range(n_primes - 1, -1, -1):
        if time.time() > deadline:
            break
        p = primes_list[idx]
        if p <= 2:
            continue
        
        k = _check_run(p, small_primes, max_k=200, deadline=deadline)
        checked += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we still have time, try larger primes beyond sieve_limit
    if time.time() < deadline - 10.0:
        large_start = sieve_limit + 1
        large_end = min(10_000_000, int((deadline - time.time()) * 500))
        
        tested_large = 0
        candidate = large_start if large_start % 2 != 0 else large_start + 1
        while candidate < large_end and time.time() < deadline:
            if _is_prime(candidate):
                k = _check_run(candidate, small_primes, max_k=200, deadline=deadline)
                tested_large += 1
                if k > best_k:
                    best_k = k
                    best_p = candidate
            candidate += 2
            if tested_large > 500:
                break
    
    # Ensure we have a valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
