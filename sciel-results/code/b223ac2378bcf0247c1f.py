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

def _find_prime_near(x, small_primes):
    """Find the nearest prime to x (searching outward). Returns None if not found within bounds."""
    if x < 2:
        return None
    if x <= 1000000:
        # Use sieve for small ranges
        for d in range(0, 1000):
            for candidate in [x + d, x - d]:
                if candidate >= 2 and _is_prime(candidate):
                    return candidate
        return None
    # For large x, search outward
    for d in range(0, 10000):
        for candidate in [x + d, x - d]:
            if candidate >= 2 and candidate <= 10**9 and _is_prime(candidate):
                return candidate
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Precompute small primes for trial division
    small_primes = _sieve_primes(1_000_000)
    
    best_k = 0
    best_p = 7  # Fallback: p=7 gives k=1
    
    # Structural change: Random sampling across [10^6, 10^9] with pre-filtering
    # Long smooth runs are statistically more likely for large p because
    # p^2 + i has factors that are less likely to exceed p.
    
    # Phase 1: Quick check of a few known small primes for baseline
    for p in [7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        k = _check_run(p, small_primes, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Random sampling in [10^6, 10^9]
    # Use random primes from this range. For each candidate, do a quick pre-filter:
    # check if p^2+1 has all factors <= p before doing full check.
    
    rng = random.Random(42)
    checked = 0
    
    while time.time() < deadline:
        # Generate random candidate in [10^6, 10^9]
        x = rng.randint(10**6, 10**9)
        
        # Find nearest prime
        p = _find_prime_near(x, small_primes)
        if p is None or p < 1000 or p > 10**9:
            continue
        
        # Quick pre-filter: check if p^2+1 is p-smooth
        # If not, skip this candidate (saves time on full check)
        val1 = p * p + 1
        # Quick trial division by small primes
        temp = val1
        for sp in small_primes:
            if sp > p:
                break
            if temp % sp == 0:
                while temp % sp == 0:
                    temp //= sp
        if temp > 1:
            # Check if remaining temp is prime and <= p
            if _is_prime(temp):
                if temp > p:
                    continue  # p^2+1 has a prime factor > p, skip
            else:
                # Need to factor temp; if it has a factor > p, skip
                # Quick check: if temp > p*p, it might have a large factor
                # But let's just do the full check for now
                pass
        
        # Do full check
        k = _check_run(p, small_primes, max_k=200, deadline=deadline)
        checked += 1
        
        if k > best_k:
            best_k = k
            best_p = p
        
        # If we found a very good result, try to refine around it
        if best_k >= 10 and time.time() < deadline - 20:
            # Try nearby primes
            for delta in [-1, 1, -2, 2, -3, 3, -5, 5, -7, 7]:
                np_ = p + delta
                if np_ >= 1000 and np_ <= 10**9 and _is_prime(np_):
                    k2 = _check_run(np_, small_primes, max_k=200, deadline=deadline)
                    if k2 > best_k:
                        best_k = k2
                        best_p = np_
                if time.time() > deadline:
                    break
    
    # Ensure we have a valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
