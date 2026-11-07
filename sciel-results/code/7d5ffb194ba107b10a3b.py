"""Search for M with many k such that k*sigma(k)=M."""
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    # Best known fallback
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Sieve sigma up to a limit
    # k*sigma(k) <= 10^18, and sigma(k) is typically O(k log log k)
    # So k can be up to roughly 10^18 / (k log log k) => k ~ 10^9 / log log k
    # But for finding collisions, we want to go as far as possible
    # Let's try limit = 5*10^6 first, then extend if time permits
    
    for limit in [10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000, 200_000, 100_000]:
        if time.time() > deadline:
            break
        
        # Sieve sigma
        sigma = [0] * (limit + 1)
        for i in range(1, limit + 1):
            for j in range(i, limit + 1, i):
                sigma[j] += i
        
        # Compute v(k) = k * sigma(k) and bucket
        buckets = defaultdict(list)
        for k in range(1, limit + 1):
            v = k * sigma[k]
            if v <= 10**18:
                buckets[v].append(k)
        
        # Find the bucket with most entries
        max_count = 0
        best_v = None
        best_ks = []
        for v, ks in buckets.items():
            if len(ks) > max_count:
                max_count = len(ks)
                best_v = v
                best_ks = ks
        
        if max_count >= len(best_k):
            best_M = best_v
            best_k = best_ks
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Verify with exact factorization
    def sigma_exact(n):
        """Compute sigma(n) by trial division."""
        if n <= 0:
            return 0
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
            result *= (temp + 1)
        return result
    
    verified = []
    for k in best_k:
        if k * sigma_exact(k) == best_M:
            verified.append(k)
    
    # Ensure distinct
    verified = list(set(verified))
    
    if len(verified) >= len(best_k) and len(verified) > 0:
        return {'M': int(best_M), 'k': [int(k) for k in verified]}
    else:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
