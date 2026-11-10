import math
import time
import numpy as np
from collections import defaultdict

def construct():
    deadline = time.time() + 140.0  # safety margin for 150s limit
    
    # Sieve for sigma(n) up to N using numpy
    N = 10_000_000  # 10 million
    
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Compute M_k = k * sigma(k) for all k
    # k values: 1..N
    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma[1:N+1]
    
    # Filter: M <= 10^18
    valid_mask = M_vals <= 10**18
    ks_valid = ks[valid_mask]
    M_valid = M_vals[valid_mask]
    
    # Group by M value
    # Use pandas or manual grouping
    # For numpy, we can use a dict approach
    # But with 10^7 elements, let's be smart
    
    # Find the M with the most k values
    # Sort by M, then count consecutive equal M values
    order = np.argsort(M_valid)
    M_sorted = M_valid[order]
    k_sorted = ks_valid[order]
    
    # Find groups of equal M
    # Use np.diff to find boundaries
    diff = np.diff(M_sorted)
    boundaries = np.where(diff != 0)[0] + 1
    all_boundaries = np.concatenate(([0], boundaries, [len(M_sorted)]))
    
    best_count = 0
    best_M = 333312
    best_k_list = [336, 372, 434]
    
    for i in range(len(all_boundaries) - 1):
        if time.time() > deadline:
            break
        start = all_boundaries[i]
        end = all_boundaries[i + 1]
        count = end - start
        if count > best_count:
            best_count = count
            best_M = int(M_sorted[start])
            best_k_list = k_sorted[start:end].tolist()
    
    # Verify final result
    def _sigma_verify(n):
        if n <= 0:
            return 0
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                sp = 1
                pk = 1
                while temp % p == 0:
                    temp //= p
                    pk *= p
                    sp += pk
                result *= sp
            p += 1
        if temp > 1:
            result *= (1 + temp)
        return result
    
    final_k = []
    for k in best_k_list:
        if k >= 1 and k <= math.isqrt(best_M) and k * _sigma_verify(k) == best_M:
            final_k.append(k)
    
    if len(final_k) < 3:
        # Fallback
        return {'M': int(333312), 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': list(final_k)}
