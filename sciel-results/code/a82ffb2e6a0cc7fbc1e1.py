import math
import time
from collections import defaultdict

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def _divisors(n):
    """Return all divisors of n."""
    divs = []
    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill

    def _fallback():
        # Try several known M values with multiple solutions
        candidates = [
            (333312, [336, 372, 434]),
            (1, [1]),
        ]
        for M, ks in candidates:
            ok = []
            for k in ks:
                if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                    ok.append(k)
            if len(ok) >= 1:
                return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    # Strategy: search for M values with many preimages
    # Generate candidate M values that are likely to have many k such that k*sigma(k) = M
    
    best_M = None
    best_sols = None
    
    # Try M values that are products of small primes
    # or M values in a range where we expect many collisions
    
    # First, use the sieve approach to find a good M
    import numpy as np
    N = 20_000_000
    
    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 40.0:
                break
    except Exception:
        return _fallback()
    
    if time.time() > deadline - 35.0:
        return _fallback()
    
    # Build map from M -> list of k
    M_to_ks = defaultdict(list)
    sigma_list = sigma.tolist()
    for k in range(1, N + 1):
        M_val = k * sigma_list[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
        if time.time() > deadline - 25.0:
            break
    
    if time.time() > deadline - 20.0:
        return _fallback()
    
    # Pick the M with the most valid k's (k <= sqrt(M))
    for M_val, ks in M_to_ks.items():
        if len(ks) < 2:
            continue
        limit = math.isqrt(M_val)
        valid = [k for k in ks if k <= limit]
        if best_sols is None or len(valid) > len(best_sols):
            best_sols = valid
            best_M = M_val
        if time.time() > deadline - 10.0:
            break
    
    if best_sols is None or len(best_sols) < 2:
        return _fallback()
    
    # Now, for the best M, try to find additional k values
    # by checking divisors of M that might satisfy k*sigma(k) = M
    M_best = best_M
    divs = _divisors(M_best)
    for k in divs:
        if k <= math.isqrt(M_best) and k not in best_sols:
            if k * _sigma_slow(k) == M_best:
                best_sols.append(k)
        if time.time() > deadline - 5.0:
            break
    
    # Final exact verification
    verified = []
    for k in best_sols:
        if 1 <= k <= math.isqrt(M_best):
            if k * _sigma_slow(k) == M_best:
                verified.append(k)
    
    if not verified:
        return _fallback()
    
    verified = sorted(set(verified))
    return {'M': int(M_best), 'k': [int(x) for x in verified]}
