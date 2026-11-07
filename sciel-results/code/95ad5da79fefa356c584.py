"""Rewritten program: search for M with many k such that k*sigma(k)=M."""
import math
import time
from collections import defaultdict

def isqrt(n):
    return math.isqrt(n)

def sieve_smallest_prime(limit):
    """Return spf array: spf[i] = smallest prime factor of i."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def sigma_fast(n, spf):
    """Compute sigma(n) using smallest-prime-factor table. n must be < len(spf)."""
    if n <= 1:
        return n
    result = 1
    m = n
    while m > 1:
        p = spf[m]
        power = 1
        sigma_part = 1
        while m % p == 0:
            m //= p
            power *= p
            sigma_part += power
        result *= sigma_part
    return result

def sigma_general(n):
    """Compute sigma(n) for arbitrary n using trial division."""
    if n <= 1:
        return n
    result = 1
    d = 2
    nn = n
    while d * d <= nn:
        if nn % d == 0:
            power = 1
            sigma_part = 1
            while nn % d == 0:
                nn //= d
                power *= d
                sigma_part += power
            result *= sigma_part
        d += 1 if d == 2 else 2
    if nn > 1:
        result *= (1 + nn)
    return result

def construct():
    deadline = time.time() + 115  # 115s safety margin (limit 150s)
    
    # Strategy: compute M = k*sigma(k) for many k, group by M,
    # find M with most distinct k values (k <= sqrt(M)).
    
    # Precompute SPF up to a large limit
    SPF_LIMIT = 10_000_000
    spf = sieve_smallest_prime(SPF_LIMIT)
    
    # Build M -> set of k
    m_to_k = defaultdict(set)
    
    # First pass: k from 1 to SPF_LIMIT using fast sigma
    for k in range(1, SPF_LIMIT + 1):
        if time.time() > deadline - 5:
            break
        s = sigma_fast(k, spf)
        M = k * s
        if M > 10**18:
            continue
        if k <= isqrt(M):
            m_to_k[M].add(k)
    
    # Find current best
    best_M = None
    best_count = 0
    for M, k_set in m_to_k.items():
        if len(k_set) > best_count:
            best_count = len(k_set)
            best_M = M
    
    # If we have time, extend search to larger k values
    if time.time() < deadline - 20:
        K_LOW = SPF_LIMIT + 1
        K_HIGH = 20_000_000
        
        for k in range(K_LOW, K_HIGH + 1):
            if time.time() > deadline - 10:
                break
            s = sigma_general(k)
            M = k * s
            if M > 10**18:
                continue
            if k <= isqrt(M):
                m_to_k[M].add(k)
        
        # Re-evaluate best
        for M, k_set in m_to_k.items():
            if len(k_set) > best_count:
                best_count = len(k_set)
                best_M = M
    
    if best_M is None:
        best_M = 333312
        best_k_list = [336, 372, 434]
    else:
        best_k_list = sorted(m_to_k[best_M])
    
    # Ensure types are Python int
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    # Deduplicate and sort
    best_k_list = sorted(set(best_k_list))
    
    # Truncate to 1000
    best_k_list = best_k_list[:1000]
    
    return {'M': best_M, 'k': best_k_list}
