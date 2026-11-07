"""Rewritten program: search for M with many k such that k*sigma(k)=M."""
import math
import time
import random
from collections import defaultdict

# ---------- helpers ----------

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
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

def sigma_fast(n, spf):
    """Compute sigma(n) using smallest-prime-factor table. n must be < len(spf)."""
    if n <= 0:
        return 0
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
    if n <= 0:
        return 0
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
        d += 1
    if nn > 1:
        result *= (1 + nn)
    return result

def verify_solution(M, k, spf):
    """Check k*sigma(k) == M. If k < len(spf) use fast, else general."""
    if k < 0 or M < 0:
        return False
    if k < len(spf):
        s = sigma_fast(k, spf)
    else:
        s = sigma_general(k)
    return k * s == M

# ---------- main construct ----------

def construct():
    deadline = time.time() + 110  # 110s safety margin (limit 120s)
    
    # Strategy: search over candidate k values, compute M = k*sigma(k),
    # group by M, and return the M with the most distinct k (k <= sqrt(M)).
    
    # Precompute smallest-prime-factor table up to a reasonable limit
    SPF_LIMIT = 5_000_000
    spf = sieve_smallest_prime(SPF_LIMIT)
    
    # We'll collect: M -> set of k
    # Focus on k up to some bound to keep computation tractable
    # For M <= 1e18, k <= sqrt(M) <= ~1e9, but sigma(k) grows, so k*sigma(k) grows fast.
    # Let's search k in [1, K_MAX] and track M values.
    
    K_MAX = min(SPF_LIMIT - 1, 2_000_000)  # cap to stay within SPF table
    
    # Build M -> list of k
    m_to_k = defaultdict(list)
    
    for k in range(1, K_MAX + 1):
        if time.time() > deadline:
            break
        s = sigma_fast(k, spf)
        M = k * s
        if M > 10**18:
            continue
        if k <= isqrt(M):  # must satisfy k <= sqrt(M)
            m_to_k[M].append(k)
    
    # Find best M (most k values, all distinct and valid)
    best_M = None
    best_k_list = []
    
    for M, k_list in m_to_k.items():
        if len(k_list) > len(best_k_list):
            # Verify all
            valid_ks = []
            for k in k_list:
                if verify_solution(M, k, spf):
                    valid_ks.append(k)
            if len(valid_ks) > len(best_k_list):
                best_M = M
                best_k_list = valid_ks
    
    if best_M is None:
        # Fallback to known solution
        best_M = 333312
        best_k_list = [336, 372, 434]
    
    # If we have time, try a second pass with a different range or smarter search
    if time.time() < deadline - 10:
        # Try searching in a higher range for k (using general sigma for large k)
        # This is slower but might find more solutions
        K_LOW = K_MAX + 1
        K_HIGH = min(5_000_000, K_MAX + 1_500_000)
        
        for k in range(K_LOW, K_HIGH + 1):
            if time.time() > deadline - 5:
                break
            s = sigma_general(k)
            M = k * s
            if M > 10**18:
                continue
            if k <= isqrt(M):
                if M in m_to_k:
                    if k not in m_to_k[M]:
                        m_to_k[M].append(k)
                else:
                    m_to_k[M] = [k]
        
        # Re-evaluate best
        for M, k_list in m_to_k.items():
            if len(k_list) > len(best_k_list):
                valid_ks = []
                for k in k_list:
                    if verify_solution(M, k, spf):
                        valid_ks.append(k)
                if len(valid_ks) > len(best_k_list):
                    best_M = M
                    best_k_list = valid_ks
    
    # Ensure types are Python int
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    # Deduplicate and sort
    best_k_list = sorted(set(best_k_list))
    
    # Truncate to 1000
    best_k_list = best_k_list[:1000]
    
    return {'M': best_M, 'k': best_k_list}
