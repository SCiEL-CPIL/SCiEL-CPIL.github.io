"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def _sigma(n):
    """Compute sigma(n) = sum of divisors of n via trial division."""
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def _sigma_sieve(lo, hi):
    """Compute sigma(k) for all k in [lo, hi] using a sieve approach with NumPy.
    Returns a numpy array sigma_vals where sigma_vals[i] = sigma(lo+i)."""
    n = hi + 1
    # Smallest prime factor sieve using numpy for speed
    spf = np.arange(n, dtype=np.int64)
    limit = int(math.isqrt(n))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            spf[start:n:i] = np.where(spf[start:n:i] == spf[start:n:i], i, spf[start:n:i])
            # More efficient: only set where spf[j] == j
            for j in range(i * i, n, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma for each number using spf
    sigma_vals = np.ones(n, dtype=np.int64)
    sigma_vals[0] = 0
    if n > 1:
        sigma_vals[1] = 1
    
    for k in range(2, n):
        temp = k
        prod = 1
        while temp > 1:
            p = int(spf[temp])
            pk = 1
            term = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            prod *= term
        sigma_vals[k] = prod
    
    return sigma_vals[lo:hi+1]

def _sigma_sieve_fast(lo, hi):
    """Faster sigma computation using multiplicative property with SPF."""
    n = hi + 1
    # Build SPF
    spf = list(range(n))
    limit = int(math.isqrt(n))
    for i in range(2, limit + 1):
        if spf[i] == i:
            for j in range(i * i, n, i):
                if spf[j] == j:
                    spf[j] = i
    
    # sigma[1] = 1
    sigma = [1] * n
    sigma[0] = 0
    
    for k in range(2, n):
        p = spf[k]
        if k // p == 0 or spf[k // p] != p:
            # k = p^1 * m where p does not divide m
            sigma[k] = sigma[k // p] * (1 + p)
        else:
            # k = p^a * m where p divides m
            m = k
            pk = 1
            while m % p == 0:
                m //= p
                pk *= p
            sigma[k] = sigma[m] * (pk * (p + 1) - 1) // (p - 1)
    
    return sigma[lo:hi+1]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s safety margin (15s buffer)
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: sieve k*sigma(k) over large ranges, group by M value
    # k*sigma(k) ~ O(k^2 log k), so for M <= 10^18, k can be quite large
    # But we're limited by memory and time
    
    # Use the fast SPF-based sigma computation
    # Process in chunks to manage memory
    chunk_size = 2000000  # 2M per chunk
    
    # Estimate how far we can search based on available time
    # Each chunk of 2M takes roughly 2-5 seconds with the fast method
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    # Conservative estimate: ~3s per 2M chunk
    estimated_chunks = int(remaining / 3.0)
    max_k_search = min(20000000, 2 + estimated_chunks * chunk_size)
    
    print(f"Estimated max_k_search: {max_k_search}, remaining: {remaining:.1f}s")
    
    for lo in range(2, max_k_search + 1, chunk_size):
        if time.time() > deadline - 10:
            break
        
        hi = min(lo + chunk_size - 1, max_k_search)
        
        # Compute sigma for this range
        sigma_vals = _sigma_sieve_fast(lo, hi)
        
        groups = defaultdict(list)
        for i, k in enumerate(range(lo, hi + 1)):
            s = sigma_vals[i]
            M = k * s
            if M <= 10**18 and k <= math.isqrt(M):
                groups[M].append(k)
        
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_M = M
                best_k = list(ks)
                best_count = len(ks)
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # Second pass: extend best_M by searching nearby k values
    if time.time() < deadline - 20 and best_count > 3:
        existing_ks = set(best_k)
        min_k = min(existing_ks) if existing_ks else 1
        max_k = max(existing_ks) if existing_ks else 100
        search_lo = max(2, min_k - 100000)
        search_hi = min(max_k_search, max_k + 100000)
        
        if search_hi > search_lo:
            chunk_lo = search_lo
            while chunk_lo < search_hi:
                if time.time() > deadline - 10:
                    break
                chunk_hi = min(chunk_lo + chunk_size - 1, search_hi)
                sigma_vals = _sigma_sieve_fast(chunk_lo, chunk_hi)
                for i, k in enumerate(range(chunk_lo, chunk_hi + 1)):
                    s = sigma_vals[i]
                    M = k * s
                    if M == best_M and k <= math.isqrt(M) and k not in existing_ks:
                        best_k.append(k)
                        existing_ks.add(k)
                chunk_lo = chunk_hi + 1
            
            best_count = len(best_k)
    
    # Third pass: if time permits, extend to larger k values
    if time.time() < deadline - 15:
        extended_max = min(30000000, int((deadline - time.time()) * 100000))
        if extended_max > max_k_search:
            for lo in range(max_k_search + 1, extended_max + 1, chunk_size):
                if time.time() > deadline - 10:
                    break
                hi = min(lo + chunk_size - 1, extended_max)
                sigma_vals = _sigma_sieve_fast(lo, hi)
                groups = defaultdict(list)
                for i, k in enumerate(range(lo, hi + 1)):
                    s = sigma_vals[i]
                    M = k * s
                    if M <= 10**18 and k <= math.isqrt(M):
                        groups[M].append(k)
                for M, ks in groups.items():
                    if len(ks) > best_count:
                        best_M = M
                        best_k = list(ks)
                        best_count = len(ks)
    
    # Final validation and return
    valid_ks = []
    for k in best_k:
        if isinstance(k, int) and not isinstance(k, bool):
            if 1 <= k <= math.isqrt(best_M):
                s = _sigma(k)
                if k * s == best_M:
                    valid_ks.append(k)
    
    valid_ks = list(set(valid_ks))
    valid_ks.sort()
    
    if len(valid_ks) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in valid_ks]}
# EVOLVE-BLOCK-END
