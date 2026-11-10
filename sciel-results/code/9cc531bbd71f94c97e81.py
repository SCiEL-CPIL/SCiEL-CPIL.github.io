"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _sigma_sieve_numpy(limit):
    """Compute sigma(k) for all k up to limit using numpy."""
    sigma = np.zeros(limit + 1, dtype=np.int64)
    for i in range(1, limit + 1):
        sigma[i::i] += i
    return sigma

def _find_best_m(sigma, limit, time_budget=None):
    """Group k by M = k*sigma(k), find M with most k values."""
    # Compute M = k * sigma(k) for all k
    ks = np.arange(1, limit + 1, dtype=np.int64)
    Ms = ks * sigma[1:]
    
    # Filter to M <= 10^18
    valid = Ms <= 10**18
    Ms_valid = Ms[valid]
    ks_valid = ks[valid]
    
    if len(Ms_valid) == 0:
        return None, []
    
    # Sort by M value to group
    sort_idx = np.argsort(Ms_valid)
    Ms_sorted = Ms_valid[sort_idx]
    ks_sorted = ks_valid[sort_idx]
    
    # Find groups of equal M
    changes = np.where(np.diff(Ms_sorted) != 0)[0] + 1
    group_starts = np.concatenate([[0], changes])
    group_ends = np.concatenate([changes, [len(Ms_sorted)]])
    
    best_m = None
    best_ks = []
    best_count = 0
    
    for start, end in zip(group_starts, group_ends):
        count = end - start
        if count > best_count:
            best_count = count
            best_m = int(Ms_sorted[start])
            best_ks = [int(k) for k in ks_sorted[start:end]]
    
    return best_m, best_ks

def _sigma_by_factorization(n):
    """Compute sigma(n) by exact factorization."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power_sum = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                power_sum += power
            result *= power_sum
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _verify_solution(k, M):
    """Verify that k * sigma(k) == M using exact factorization."""
    if k < 1 or k > math.isqrt(M):
        return False
    sigma_k = _sigma_by_factorization(k)
    return k * sigma_k == M

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with known good answer
    best_m = 333312
    best_ks = [336, 372, 434]
    
    # Try increasing limits with numpy-based sieve
    limits_to_try = [200000, 500000, 1000000, 2000000, 4000000, 8000000]
    
    for limit in limits_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # need time for grouping and verification
            break
        
        try:
            sigma = _sigma_sieve_numpy(limit)
            m, ks = _find_best_m(sigma, limit)
            
            if m is not None and len(ks) > len(best_ks):
                best_m = m
                best_ks = ks
                
            if time.time() - start_time > deadline - 10:
                break
                
        except MemoryError:
            break
        except Exception:
            break
    
    # Verify the best candidates using exact factorization
    sqrt_m = math.isqrt(best_m)
    verified_ks = []
    for k in best_ks[:1000]:
        if 1 <= k <= sqrt_m:
            if _verify_solution(k, best_m):
                verified_ks.append(k)
    
    if not verified_ks:
        # Fallback to known good answer
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_m), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
