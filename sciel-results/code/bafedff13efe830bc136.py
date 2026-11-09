import time
import numpy as np
from math import isqrt, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values - larger N gives more chances for collisions
    N_candidates = [2 * 10**6, 5 * 10**6, 10 * 10**6, 20 * 10**6, 40 * 10**6, 80 * 10**6]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break
        
        # Compute sigma using vectorized NumPy sieve (sum of divisors)
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break

        # Find M with most k using vectorized operations
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        
        # Filter M <= 10^18
        mask = M_arr <= 10**18
        M_valid = M_arr[mask]
        k_valid = k_arr[mask]
        
        if len(M_valid) == 0:
            continue
        
        # Find M with most k using np.unique
        unique_M, inverse = np.unique(M_valid, return_inverse=True)
        counts = np.bincount(inverse)
        max_count = int(counts.max())
        
        if max_count <= len(best_k):
            continue
        
        # Find all M that achieve max_count
        best_Ms = unique_M[counts == max_count]
        
        # Pick the one with the most potential for additional solutions
        # Prefer smaller M (more room for larger k) or the first one
        current_best_M = int(best_Ms[0])
        
        # Get all k for this M within our enumeration range
        current_best_k_list = [int(k) for k in k_valid[M_valid == current_best_M]]
        
        if len(current_best_k_list) > len(best_k):
            best_M = current_best_M
            best_k = current_best_k_list
    
    # Refinement: try to find additional k values for the best M
    # by searching beyond the enumeration range
    if time.time() - start_time < deadline - 5:
        M = best_M
        sqrt_M = isqrt(M)
        existing_ks = set(best_k)
        
        # Try to find more k values by checking divisors of M
        # k must divide M and k <= sqrt(M)
        # Generate candidate k values from divisors of M
        candidates_k = set()
        
        # Get all divisors of M
        divisors = set()
        for i in range(1, isqrt(M) + 1):
            if M % i == 0:
                divisors.add(i)
                divisors.add(M // i)
        
        # Filter to k <= sqrt(M) and not already found
        for d in divisors:
            if d <= sqrt_M and d not in existing_ks:
                candidates_k.add(d)
        
        # Check each candidate k
        for k in sorted(candidates_k):
            if time.time() - start_time > deadline - 2:
                break
            # Compute sigma(k) exactly
            sigma_k = _sigma_exact(k)
            if k * sigma_k == M:
                best_k.append(k)
                existing_ks.add(k)
        
        # Sort best_k
        best_k = sorted(best_k)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}


def _sigma_exact(n):
    """Compute sigma(n) - sum of divisors of n - using trial division."""
    if n <= 1:
        return n
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            power = 1
            while temp % d == 0:
                temp //= d
                exp += 1
                power *= d
            result *= (power * d - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)
    return result
