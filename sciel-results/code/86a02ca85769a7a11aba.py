# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
import numpy as np
from collections import defaultdict

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 100000
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

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _find_k_for_M(M):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    """
    sqrtM = isqrt(M)
    results = []
    factors = _factorize(M)
    divisors = _all_divisors_from_factors(factors)
    
    for k in divisors:
        if k > sqrtM:
            continue
        if k < 1:
            continue
        s = _sigma(k)
        if k * s == M:
            results.append(k)
    return results

def _compute_sigma_sieve(N):
    """
    Compute sigma(n) for n=1..N using a sieve.
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    return sigma

def _smooth_search(primes, max_exp, deadline):
    """
    Generate k values that are smooth (using only given primes) with bounded exponents.
    For each, compute M = k * sigma(k) and group collisions.
    """
    collision_map = defaultdict(list)
    count = 0
    
    def _recurse(idx, current_k):
        nonlocal count
        if time.time() > deadline:
            return
        if idx == len(primes):
            if current_k > 1 and current_k <= 10**18:
                s = _sigma(current_k)
                M = current_k * s
                if M <= 10**18:
                    collision_map[M].append(current_k)
            return
        p = primes[idx]
        val = current_k
        for e in range(max_exp + 1):
            if time.time() > deadline:
                return
            _recurse(idx + 1, val)
            val *= p
            if val > 10**18:
                break
    
    _recurse(0, 1)
    return collision_map

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer
    
    # Known fallback
    fallback = {'M': 333312, 'k': [336, 372, 434]}
    
    # Phase 1: Sieve-based collision search
    best_count = 3
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Try sieve with increasing limits
    sieve_limits = [2_000_000, 5_000_000, 10_000_000]
    
    for L in sieve_limits:
        if time.time() > deadline - 30:
            break
        try:
            sigma = _compute_sigma_sieve(L)
        except Exception:
            continue
        
        if time.time() > deadline - 25:
            break
            
        k_vals = np.arange(1, L + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:L+1]
        
        mask = M_vals <= 10**18
        M_vals = M_vals[mask]
        k_vals = k_vals[mask]
        
        if len(M_vals) == 0:
            continue
            
        sorted_indices = np.argsort(M_vals)
        sorted_M = M_vals[sorted_indices]
        sorted_k = k_vals[sorted_indices]
        
        n = len(sorted_M)
        i = 0
        while i < n:
            if time.time() > deadline - 20:
                break
            j = i + 1
            while j < n and sorted_M[j] == sorted_M[i]:
                j += 1
                
            if j - i > best_count:
                M_val = int(sorted_M[i])
                group_k = sorted_k[i:j]
                
                sqrt_M = isqrt(M_val)
                valid_ks = group_k[group_k <= sqrt_M]
                
                if len(valid_ks) > best_count:
                    best_count = len(valid_ks)
                    best_M = M_val
                    best_ks = valid_ks.tolist()
                    
            i = j
        
        if time.time() > deadline - 20:
            break
    
    # Phase 2: Smooth number search to find more collisions
    if time.time() < deadline - 10:
        # Use small primes for smooth numbers
        smooth_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        # Try different exponent bounds
        for max_exp in [3, 4, 5]:
            if time.time() > deadline - 5:
                break
            
            smooth_map = _smooth_search(smooth_primes[:min(15, len(smooth_primes))], max_exp, deadline - 5)
            
            for M_val, ks_list in smooth_map.items():
                if len(ks_list) <= best_count:
                    continue
                sqrt_M = isqrt(M_val)
                valid_ks = [k for k in ks_list if k <= sqrt_M]
                if len(valid_ks) > best_count:
                    best_count = len(valid_ks)
                    best_M = M_val
                    best_ks = valid_ks
    
    # Phase 3: For the best M found, try to find additional k values
    if time.time() < deadline - 5:
        additional = _find_k_for_M(best_M)
        if len(additional) > best_count:
            best_ks = additional
            best_count = len(best_ks)
    
    # Verify the best result
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    # Ensure distinct
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        return fallback
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
