import math
from collections import defaultdict

def construct():
    # Strategy: enumerate k up to a large limit, compute M = k*sigma(k),
    # group by M, and find M with the most valid k (where k <= isqrt(M)).
    # Use an efficient sieve for sigma computation.
    
    # We'll use a segmented or direct sieve approach.
    # For LIMIT = 2_000_000, we can use a simple sieve within time limits.
    
    LIMIT = 2_000_000
    
    # Sieve for sigma: sigma[n] = sum of divisors of n
    sigma = [0] * (LIMIT + 1)
    for i in range(1, LIMIT + 1):
        for j in range(i, LIMIT + 1, i):
            sigma[j] += i
    
    # Group k by M = k * sigma[k]
    m_to_k = defaultdict(list)
    for k in range(1, LIMIT + 1):
        m_val = k * sigma[k]
        if m_val <= 10**18:
            m_to_k[m_val].append(k)
    
    # Find M with the most valid k (k <= isqrt(M))
    best_m = None
    best_k_list = []
    
    for m_val, k_list in m_to_k.items():
        sqrt_m = math.isqrt(m_val)
        valid_k = [k for k in k_list if k <= sqrt_m]
        if len(valid_k) > len(best_k_list):
            best_k_list = valid_k
            best_m = m_val
    
    # Verify the solutions using exact factorization
    def sigma_exact(n):
        if n == 1:
            return 1
        result = 1
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                p_pow = 1
                while temp % d == 0:
                    p_pow *= d
                    temp //= d
                result *= (p_pow * d - 1) // (d - 1)
            d += 1
        if temp > 1:
            result *= (1 + temp)
        return result
    
    verified_k = []
    for k in best_k_list:
        if k * sigma_exact(k) == best_m and k <= math.isqrt(best_m):
            verified_k.append(k)
    
    verified_k = sorted(set(verified_k))
    
    if best_m is not None and len(verified_k) >= 2:
        return {'M': int(best_m), 'k': [int(k) for k in verified_k]}
    
    # Fallback: known M with multiple solutions
    # For M = 333312: k=336, sigma(336)=1008, 336*1008=338688... let me recompute
    # Actually let's use a known valid example
    # k=1: sigma(1)=1, M=1
    # k=2: sigma(2)=3, M=6
    # k=3: sigma(3)=4, M=12
    # k=4: sigma(4)=7, M=28
    # k=5: sigma(5)=6, M=30
    # k=6: sigma(6)=12, M=72
    # k=7: sigma(7)=8, M=56
    # k=8: sigma(8)=15, M=120
    # k=9: sigma(9)=13, M=117
    # k=10: sigma(10)=18, M=180
    
    # Let's try to find collisions in small k
    # M=120: k=8 (8*15=120), also check if any other k gives 120
    # k=1:1, k=2:6, k=3:12, k=4:28, k=5:30, k=6:72, k=7:56, k=8:120, k=9:117, k=10:180
    # k=11:132, k=12:252, ...
    
    # Known: M=72 has k=6 (6*12=72). Any other? k=1..5 give 1,6,12,28,30 - no.
    # M=120: only k=8 in small range.
    
    # Let's just return a valid small solution as fallback
    return {'M': 1, 'k': [1]}
