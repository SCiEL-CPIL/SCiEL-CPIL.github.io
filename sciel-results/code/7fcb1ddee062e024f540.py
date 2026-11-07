"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def _sigma(n):
    """Compute sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p**(exp+1) - 1) // (p - 1)
        p += 1
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

def construct():
    import time
    deadline = time.time() + 140  # 10s safety margin
    
    # Build mapping from M = k*sigma(k) to list of k
    m_to_ks = defaultdict(list)
    
    k_limit = 200000
    for k in range(1, k_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Find M with most k values (that also satisfy k <= isqrt(M))
    best_count = 0
    best_M = 333312
    best_ks = [336, 372, 434]
    
    for M, ks in m_to_ks.items():
        if len(ks) <= best_count:
            continue
        isq = math.isqrt(M)
        valid_ks = [k for k in ks if k <= isq]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_ks = valid_ks
    
    # Extend search if time permits
    if time.time() < deadline - 5:
        k_limit2 = 500000
        for k in range(k_limit + 1, k_limit2 + 1):
            if time.time() > deadline:
                break
            s = _sigma(k)
            M = k * s
            if M <= 10**18:
                m_to_ks[M].append(k)
        
        # Re-check all M
        for M, ks in m_to_ks.items():
            if len(ks) <= best_count:
                continue
            isq = math.isqrt(M)
            valid_ks = [k for k in ks if k <= isq]
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # Final verification
    isq = math.isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= isq:
            s = _sigma(k)
            if k * s == best_M:
                final_ks.append(k)
    
    if not final_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
# EVOLVE-BLOCK-END
