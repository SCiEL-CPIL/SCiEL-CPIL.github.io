import math
from collections import defaultdict

def sigma(n):
    """Compute sum of divisors of n."""
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            while temp % p == 0:
                temp //= p
                term = term * p + 1
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    # Search for M with many k such that k*sigma(k) = M and k <= sqrt(M)
    # Group by M = k*sigma(k)
    groups = defaultdict(list)
    
    # Limit k to keep runtime reasonable
    max_k = 200000
    
    for k in range(1, max_k + 1):
        s = sigma(k)
        M = k * s
        if M <= 10**18:
            groups[M].append(k)
    
    best_M = None
    best_ks = []
    
    for M, ks in groups.items():
        sqrt_M = math.isqrt(M)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_ks):
            best_ks = valid_ks
            best_M = M
    
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
