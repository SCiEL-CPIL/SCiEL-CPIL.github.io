"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _sigma(n):
    """Compute sigma(n) exactly using trial division."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    # Factor out 2
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count > 0:
        result *= (2 ** (count + 1) - 1)
    # Factor out odd primes
    p = 3
    while p * p <= temp:
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count > 0:
            result *= (p ** (count + 1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known solution: 336, 372, 434 all satisfy k*sigma(k) = 333312
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Strategy: search for M values with many solutions k <= sqrt(M)
    # We'll try to find M values where multiple k's satisfy k*sigma(k) = M
    
    # Approach: For each candidate k, compute M = k*sigma(k), then count how many
    # other k' have the same M. We want M with many such k'.
    
    # Let's search for small k values and group by M
    # Since k <= sqrt(M), we have M >= k^2, so k*sigma(k) >= k^2, which means sigma(k) >= k
    # This is always true since sigma(k) >= k+1 for k > 1
    
    # Let's try a different approach: enumerate M values and find all k with k*sigma(k) = M
    # But M can be up to 10^18, so we need to be smart
    
    # Better approach: For small k, compute M = k*sigma(k), and find collisions
    # We want M values that appear multiple times
    
    # Let's enumerate k from 1 to some limit, compute M = k*sigma(k), and group
    # Then for each M, verify all k in the group satisfy k <= sqrt(M)
    
    # Limit: if k is large, sigma(k) grows, so M = k*sigma(k) grows quickly
    # For k <= 10^6, sigma(k) <= ~4*10^6 typically, so M <= 4*10^12
    # This is well within 10^18
    
    k_limit = 200000
    m_to_ks = {}
    
    for k in range(1, k_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M > 10**18:
            continue
        if M not in m_to_ks:
            m_to_ks[M] = []
        m_to_ks[M].append(k)
    
    # Find M with most k's where k <= sqrt(M)
    best_count = len(best_k)
    
    for M, ks in m_to_ks.items():
        if time.time() > deadline:
            break
        if len(ks) <= best_count:
            continue
        # Verify all k <= sqrt(M)
        sqrt_M = math.isqrt(M)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_k = valid_ks
    
    # If we found a better solution, return it; otherwise return the known one
    if best_count > 3:
        return {'M': int(best_M), 'k': [int(k) for k in best_k]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
