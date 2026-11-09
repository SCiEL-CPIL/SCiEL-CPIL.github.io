"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
from collections import defaultdict

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 200000
_primes = []
_is_prime = [True] * (_PRIME_LIMIT + 1)
_is_prime[0] = False
_is_prime[1] = False
for i in range(2, int(_PRIME_LIMIT**0.5) + 1):
    if _is_prime[i]:
        _is_prime[i*i::_PRIME_LIMIT+1] = [False] * len(_is_prime[i*i::_PRIME_LIMIT+1])
for i in range(2, _PRIME_LIMIT + 1):
    if _is_prime[i]:
        _primes.append(i)

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using precomputed primes."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in _primes:
        if p * p > temp:
            break
        if temp % p == 0:
            p_pow = 1
            while temp % p == 0:
                temp //= p
                p_pow *= p
            result *= (p * p_pow - 1) // (p - 1)
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline to be safe
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    collision_map = defaultdict(list)
    
    # Phase 1: Systematic search for k up to a higher limit
    search_limit = 15_000_000
    
    for k in range(1, search_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    
    # Find the M with the most solutions where k <= sqrt(M)
    for M, ks_list in collision_map.items():
        if len(ks_list) < best_count:
            continue
        sqrtM = isqrt(M)
        valid_ks = [k for k in ks_list if k <= sqrtM]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_ks = valid_ks
    
    # Phase 2: If we have time, try to extend the search
    if time.time() < deadline - 5:
        extended_limit = min(25_000_000, search_limit + 10_000_000)
        for k in range(search_limit + 1, extended_limit + 1):
            if time.time() > deadline:
                break
            s = _sigma(k)
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
        
        # Re-evaluate
        for M, ks_list in collision_map.items():
            if len(ks_list) < best_count:
                continue
            sqrtM = isqrt(M)
            valid_ks = [k for k in ks_list if k <= sqrtM]
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # Verify the best result
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    # Ensure distinct
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
