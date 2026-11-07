import math
import time
import numpy as np
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

def _find_best_in_range(N, deadline):
    """
    Computes sigma for 1..N, finds M with max collisions.
    Returns (M, ks) or (None, None) if failed.
    """
    start = time.time()
    try:
        sigma = np.zeros(N + 1, dtype=np.int32)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline:
                return None, None
        
        ks = np.arange(1, N + 1, dtype=np.int64)
        # Convert sigma to int64 for multiplication to avoid overflow in intermediate steps?
        # sigma is int32. ks is int64. Result is int64.
        M_vals = ks * sigma[1:].astype(np.int64)
        
        mask = M_vals <= 10**18
        M_valid = M_vals[mask]
        k_valid = ks[mask]
        
        if len(M_valid) == 0:
            return None, None
            
        unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        
        if len(unique_M) == 0:
            return None, None

        max_count = np.max(counts)
        if max_count < 2:
            return None, None
            
        best_idx = np.argmax(counts)
        best_M = unique_M[best_idx]
        
        target_indices = np.where(inverse == best_idx)[0]
        best_ks = k_valid[target_indices]
        
        best_ks_sorted = sorted([int(k) for k in best_ks])
        limit = math.isqrt(best_M)
        verified_ks = [k for k in best_ks_sorted if k <= limit]
        
        if not verified_ks:
            return None, None
            
        return int(best_M), verified_ks

    except Exception:
        return None, None

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin

    # Try large N first
    N_large = 100_000_000
    M_large, k_large = _find_best_in_range(N_large, deadline - 10.0)
    
    if M_large is not None:
        return {'M': int(M_large), 'k': k_large}

    # Fallback to smaller N if large one failed or timed out
    N_small = 5_000_000
    M_small, k_small = _find_best_in_range(N_small, deadline - 5.0)
    
    if M_small is not None:
        return {'M': int(M_small), 'k': k_small}

    # Hardcoded fallback
    M = 333312
    ks = [336, 372, 434]
    ok = []
    for k in ks:
        if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
            ok.append(k)
    if len(ok) >= 1:
        return {'M': int(M), 'k': [int(x) for x in ok]}
    return {'M': 1, 'k': [1]}
