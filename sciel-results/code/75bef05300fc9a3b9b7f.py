"""Rewritten program using efficient p-smooth checking for long runs after p^2."""
import math
import time

# EVOLVE-BLOCK-START

def is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.2e18."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def is_p_smooth(n, p, primes_up_to_p):
    """Check if n is p-smooth using precomputed primes up to p."""
    if n <= 1:
        return True
    for q in primes_up_to_p:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
    if n <= 1:
        return True
    # n > 1, it's a prime > last tried divisor
    # If n <= p, it's a prime factor <= p, so p-smooth
    if n <= p:
        return True
    # n > p, check if n is prime
    if is_prime_mr(n):
        return False  # prime factor > p
    # n is composite, has factors we need to check
    # Since q*q > n broke the loop, n is product of at most 2 primes
    # Both must be > sqrt(n_last) but we need them <= p
    # Actually, we stopped when q*q > n, so n is prime or product of two primes both > last q
    # Let's just factor the remainder
    return _remainder_smooth(n, p)

def _remainder_smooth(n, p):
    """Check if remaining composite n is p-smooth."""
    # n is composite, find a factor
    d = _find_factor(n)
    if d is None:
        return is_prime_mr(n) and n <= p
    m = n // d
    # Check both factors are p-smooth
    if d > p or m > p:
        # Need to factor further, but let's just check if max prime factor <= p
        factors = _factor_small(n)
        return max(factors) <= p if factors else True
    # Both d and m <= p, but they might not be prime
    # Check d is p-smooth and m is p-smooth
    # Since d and m are smaller, this should be quick
    return _is_smooth_direct(d, p) and _is_smooth_direct(m, p)

def _find_factor(n):
    """Find a non-trivial factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    # Trial division up to cube root
    limit = int(n ** (1/3)) + 2
    for d in range(5, limit, 6):
        if n % d == 0:
            return d
        if n % (d+2) == 0:
            return d+2
    # Check if n is prime
    if is_prime_mr(n):
        return None
    # Try a bit more
    for d in range(max(5, limit), min(limit + 50000, int(math.isqrt(n)) + 1)):
        if n % d == 0:
            return d
    return None

def _factor_small(n):
    """Factor n completely, assuming n is not too large."""
    factors = []
    d = 2
    while d * d <= n and d < 100000:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        if is_prime_mr(n):
            factors.append(n)
        else:
            # n is composite, try harder
            d2 = _find_factor(n)
            if d2:
                factors.extend(_factor_small(d2))
                factors.extend(_factor_small(n // d2))
            else:
                factors.append(n)
    return factors

def _is_smooth_direct(n, p):
    """Check if n is p-smooth by direct factoring (for small n)."""
    if n <= 1:
        return True
    if n <= p:
        return True  # n is prime <= p
    factors = _factor_small(n)
    return all(f <= p for f in factors) if factors else True

def construct():
    start_time = time.time()
    time_limit = 110
    
    # Precompute primes
    max_prime_search = 200000
    all_primes = sieve_primes(max_prime_search)
    
    best_k = 0
    best_p = 3
    
    def check_p(p):
        """Check k for given prime p."""
        # Get primes up to p for this check
        # Binary search for index
        lo, hi = 0, len(all_primes)
        while lo < hi:
            mid = (lo + hi) // 2
            if all_primes[mid] <= p:
                lo = mid + 1
            else:
                hi = mid
        primes_up_to_p = all_primes[:lo]
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if not is_p_smooth(n, p, primes_up_to_p):
                break
            k += 1
        return k
    
    # Phase 1: Check small primes densely
    for p in all_primes:
        if p > 5000:
            break
        if time.time() - start_time > time_limit:
            break
        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check medium primes with sampling
    if time.time() - start_time < time_limit:
        for p in all_primes:
            if p < 5000 or p > 100000:
                continue
            if time.time() - start_time > time_limit:
                break
            # Sample: check every 10th prime in this range, plus neighbors of good ones
            if (all_primes.index(p) // 10) != (all_primes.index(p) - 1) // 10 and p > 10000:
                continue
            k = check_p(p)
            if k > best_k:
                best_k = k
                best_p = p
                # Check neighbors
                idx = all_primes.index(p)
                for idx2 in range(max(0, idx-3), min(len(all_primes), idx+3)):
                    if time.time() - start_time > time_limit:
                        break
                    p2 = all_primes[idx2]
                    if p2 == p:
                        continue
                    k2 = check_p(p2)
                    if k2 > best_k:
                        best_k = k2
                        best_p = p2
    
    # Phase 3: Try some large primes
    if time.time() - start_time < time_limit - 10:
        for p in all_primes:
            if p < 100000 or p > 200000:
                continue
            if time.time() - start_time > time_limit:
                break
            if all_primes.index(p) % 5 != 0:
                continue
            k = check_p(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
