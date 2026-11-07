"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

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
    """Compute sigma(k) for all k in [lo, hi] using a sieve approach.
    Returns a list sigma_vals where sigma_vals[i] = sigma(lo+i)."""
    n = hi + 1  # we sieve up to hi
    # Smallest prime factor sieve
    spf = list(range(n))
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma for each number using its factorization
    # sigma is multiplicative: sigma(p^a) = (p^(a+1)-1)/(p-1)
    # We compute sigma(n) by factoring n via spf
    sigma_vals = [0] * (hi + 1)
    sigma_vals[0] = 0
    if hi >= 1:
        sigma_vals[1] = 1
    
    for k in range(2, hi + 1):
        # Factor k using spf
        temp = k
        prod = 1
        while temp > 1:
            p = spf[temp]
            # Compute p^a and sigma(p^a)
            pk = 1
            term = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            prod *= term
        sigma_vals[k] = prod
    
    return sigma_vals[lo:hi+1]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: use sieve to compute k*sigma(k) for large ranges, group by M
    # k*sigma(k) is roughly O(k^2) on average, so for M <= 10^18, k can be up to ~10^9
    # But sigma(k) varies, so let's explore ranges systematically
    
    # First pass: medium ranges with sieve (fast)
    # For range [lo, hi], we need hi not too large for memory (spf array of size hi+1)
    # Let's do ranges of size up to ~2M at a time
    
    max_k_search = 3000000  # up to 3 million k values
    
    # Process in chunks to manage memory
    chunk_size = 1000000
    
    for lo in range(2, max_k_search + 1, chunk_size):
        if time.time() > deadline - 10:
            break
        
        hi = min(lo + chunk_size - 1, max_k_search)
        
        # Compute sigma for this range
        sigma_vals = _sigma_sieve(lo, hi)
        
        groups = defaultdict(list)
        for i, k in enumerate(range(lo, hi + 1)):
            s = sigma_vals[i]
            M = k * s
            if M <= 10**18 and k <= math.isqrt(M):
                groups[M].append(k)
        
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_M = M
                best_k = ks
                best_count = len(ks)
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # Second pass: if time remains, try larger individual k values
    # Focus on k values that might produce the same M
    # Since k*sigma(k) ≈ c*k^2, for a given M, k ≈ sqrt(M/c)
    # Try to find M with many preimages by checking specific k values
    
    if time.time() < deadline - 20:
        # Try targeted search: for each k in a sparse set, compute M and check neighbors
        # Also try to extend the best_M found by checking nearby k values
        if best_count > 3:
            # Check a wider range around the k values that gave best_M
            existing_ks = set(best_k)
            min_k = min(existing_ks) if existing_ks else 1
            max_k = max(existing_ks) if existing_ks else 100
            search_lo = max(2, min_k - 50000)
            search_hi = min(max_k_search, max_k + 50000)
            
            if search_hi > search_lo:
                chunk_lo = search_lo
                while chunk_lo < search_hi:
                    if time.time() > deadline - 10:
                        break
                    chunk_hi = min(chunk_lo + chunk_size - 1, search_hi)
                    sigma_vals = _sigma_sieve(chunk_lo, chunk_hi)
                    for i, k in enumerate(range(chunk_lo, chunk_hi + 1)):
                        s = sigma_vals[i]
                        M = k * s
                        if M == best_M and k <= math.isqrt(M) and k not in existing_ks:
                            best_k.append(k)
                            existing_ks.add(k)
                    chunk_lo = chunk_hi + 1
                
                best_count = len(best_k)
    
    # Third pass: try even larger ranges if time permits
    if time.time() < deadline - 15 and max_k_search < 10000000:
        extended_max = min(10000000, int((deadline - time.time()) * 50000))  # rough estimate
        if extended_max > max_k_search:
            for lo in range(max_k_search + 1, extended_max + 1, chunk_size):
                if time.time() > deadline - 10:
                    break
                hi = min(lo + chunk_size - 1, extended_max)
                sigma_vals = _sigma_sieve(lo, hi)
                groups = defaultdict(list)
                for i, k in enumerate(range(lo, hi + 1)):
                    s = sigma_vals[i]
                    M = k * s
                    if M <= 10**18 and k <= math.isqrt(M):
                        groups[M].append(k)
                for M, ks in groups.items():
                    if len(ks) > best_count:
                        best_M = M
                        best_k = ks
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
