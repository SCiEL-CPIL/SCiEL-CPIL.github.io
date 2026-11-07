"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def construct():
    k_max = 10_000_000  # 10 million
    
    # Sieve for smallest prime factor
    spf = list(range(k_max + 1))
    for i in range(2, int(math.isqrt(k_max)) + 1):
        if spf[i] == i:
            for j in range(i * i, k_max + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(k) for all k using spf and multiplicative property
    sigma = [0] * (k_max + 1)
    sigma[1] = 1
    for n in range(2, k_max + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p^2 divides n: find p-power part
            mp = m
            p_pow = p
            while mp % p == 0:
                mp //= p
                p_pow *= p
            # n = p^{a+1} * mp, m = p^a * mp
            # sigma(n) = sigma(mp) * (p^{a+2}-1)/(p-1)
            # p_pow = p^{a+1}, so p^{a+2} = p_pow * p
            sigma[n] = sigma[mp] * ((p_pow * p - 1) // (p - 1))
        else:
            # gcd(p, m) = 1
            sigma[n] = sigma[m] * (p + 1)
    
    # Group by M = k * sigma(k)
    m_to_k = defaultdict(list)
    for k in range(1, k_max + 1):
        M = k * sigma[k]
        if M <= 10**18 and k <= math.isqrt(M):
            m_to_k[M].append(k)
    
    # Find best M
    best_M = None
    best_k_list = []
    for M, k_list in m_to_k.items():
        distinct_k = sorted(set(k_list))
        if len(distinct_k) > len(best_k_list):
            best_M = M
            best_k_list = distinct_k
    
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    best_k_list = best_k_list[:1000]
    return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}
# EVOLVE-BLOCK-END
