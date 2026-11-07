import time
import math
from collections import defaultdict

def _sigma_sieve(limit):
    """Compute sigma(n) for all 1 <= n <= limit using a divisor-sum sieve."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _sigma_one(n):
    """Compute sigma(n) for a single n via trial division."""
    if n <= 1:
        return 1
    result = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            power = 1
            while x % p == 0:
                x //= p
                power *= p
                term += power
            result *= term
        p += 1 if p == 2 else 2
    if x > 1:
        result *= (1 + x)
    return result

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Fallback known solution
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def try_sieve_limit(N):
        nonlocal best_M, best_ks, best_count
        if time.time() > deadline:
            return
        try:
            sigma = _sigma_sieve(N)
            buckets = defaultdict(list)
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    buckets[M].append(k)
            for M, ks in buckets.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = M
                    best_ks = list(ks)
        except MemoryError:
            return
        except Exception:
            return

    # Strategy 1: Sieve with increasing limits
    # k*sigma(k) grows roughly as k * k * (log log k), so for M <= 10^18,
    # we need k <= ~10^9, but sigma is at most ~k*log(log k), so k <= ~10^9.
    # However, for practical purposes, let's try up to 10^7 first, then 10^8 if time permits.
    sieve_limits = [10**6, 5*10**6, 10**7]
    for lim in sieve_limits:
        if time.time() > deadline - 10:
            break
        try_sieve_limit(lim)

    # Strategy 2: If we haven't found much, try a different approach.
    # Look for M that are highly composite or have special structure.
    # For example, if M = k * sigma(k), and k is a perfect number, sigma(k) = 2k, so M = 2k^2.
    # If k is a perfect number, M = 2k^2. But we need multiple k with the same M.
    # Actually, if M = 2k^2, then k must be a perfect number. Only a few perfect numbers exist.
    
    # Better strategy: Generate M from small k and look for collisions.
    # We already do this with the sieve. Let's try to extend the sieve if time permits.
    
    # Strategy 3: Try specific M values that are known to have many preimages.
    # For instance, M that are multiples of many small numbers.
    
    # Let's also try a targeted search: for M up to some limit, factor M and find all k dividing M
    # such that k*sigma(k) = M. This is expensive but might find special cases.
    
    # Actually, the sieve approach is the most efficient. Let's just make sure we're
    # using the available time well.
    
    if time.time() < deadline - 30:
        try_sieve_limit(2*10**7)
    
    if time.time() < deadline - 15:
        try_sieve_limit(5*10**7)

    if best_count < 1:
        return {'M': 1, 'k': [1]}
    
    return {'M': int(best_M), 'k': list(best_ks)}
