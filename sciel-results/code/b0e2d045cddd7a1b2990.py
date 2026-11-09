"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
from collections import defaultdict

def _sigma_single(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    if temp % 2 == 0:
        p_pow = 1
        while temp % 2 == 0:
            temp //= 2
            p_pow *= 2
        result *= (2 * p_pow - 1)
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            p_pow = 1
            while temp % f == 0:
                temp //= f
                p_pow *= f
            result *= (f * p_pow - 1) // (f - 1)
        f += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _sigma_sieve(n):
    """Compute sigma(k) for all k from 1 to n using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def _search_collisions_sieve(max_k, deadline, start_k=1):
    """
    Search for collisions in k*sigma(k) for k in [start_k, max_k] using sieve.
    Returns dict: M -> list of k values.
    """
    collision_map = defaultdict(list)
    
    # Use sieve for sigma computation
    if max_k <= 50_000_000:
        # Sieve approach: O(n log n) but very fast in practice
        sigmas = _sigma_sieve(max_k)
        for k in range(start_k, max_k + 1):
            if time.time() > deadline:
                break
            s = sigmas[k]
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
    else:
        # Fallback to individual sigma computation
        for k in range(start_k, max_k + 1):
            if time.time() > deadline:
                break
            s = _sigma_single(k)
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
    
    return collision_map

def _evaluate_collision_map(collision_map, best_count):
    """Find the M with the most valid k solutions."""
    best_M = 0
    best_ks = []
    best_c = best_count
    
    for M, ks_list in collision_map.items():
        if len(ks_list) <= best_c:
            continue
        sqrtM = isqrt(M)
        valid_ks = [k for k in ks_list if k <= sqrtM]
        if len(valid_ks) > best_c:
            best_c = len(valid_ks)
            best_M = M
            best_ks = valid_ks
    
    return best_M, best_ks, best_c

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline (150s limit - safety margin)
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    collision_map = defaultdict(list)
    
    # Phase 1: Sieve-based search up to 20M (fast)
    limit1 = 20_000_000
    t0 = time.time()
    sigmas = _sigma_sieve(limit1)
    for k in range(1, limit1 + 1):
        if time.time() > deadline:
            break
        s = sigmas[k]
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    t1 = time.time()
    
    # Evaluate after phase 1
    best_M, best_ks, best_count = _evaluate_collision_map(collision_map, best_count)
    
    # Phase 2: If time remains, extend search with individual sigma
    if time.time() < deadline - 10:
        # Continue from limit1+1 with individual sigma (slower per element)
        # Estimate how far we can go: ~10-20M more with individual sigma in remaining time
        remaining_time = deadline - time.time()
        # Individual sigma takes ~5-50us per call; estimate conservatively
        # Try to extend up to 50M total
        limit2 = min(50_000_000, limit1 + int(remaining_time * 3_000_000))  # ~3M per sec estimate
        
        for k in range(limit1 + 1, limit2 + 1):
            if time.time() > deadline:
                break
            s = _sigma_single(k)
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
        
        # Re-evaluate
        best_M, best_ks, best_count = _evaluate_collision_map(collision_map, best_count)
    
    # Verify the best result
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma_single(k) == best_M:
            verified_ks.append(k)
    
    # Ensure distinct and sorted
    verified_ks = sorted(set(verified_ks))
    
    if len(verified_ks) == 0:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
