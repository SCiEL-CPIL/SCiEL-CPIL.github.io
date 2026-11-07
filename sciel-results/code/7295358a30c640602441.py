import time
import math
import sys
from collections import defaultdict

# Pollard Rho for factorization of 64-bit integers
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = y = 2
    return d

def factorize(n):
    factors = {}
    if n % 2 == 0:
        cnt = 0
        while n % 2 == 0:
            n //= 2
            cnt += 1
        factors[2] = cnt
    p = 3
    while p * p <= n:
        if n % p == 0:
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        p += 2
    if n > 1:
        # n is prime or product of two large primes
        # Use Pollard Rho if n is large
        if n > 10**12:
            d = pollard_rho(n)
            f1 = factorize(d)
            f2 = factorize(n // d)
            for p, c in f1.items():
                factors[p] = factors.get(p, 0) + c
            for p, c in f2.items():
                factors[p] = factors.get(p, 0) + c
        else:
            factors[n] = 1
    return factors

def get_divisors(factors):
    divs = [1]
    for p, c in factors.items():
        new_divs = []
        p_power = 1
        for i in range(c + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    return divs

def sigma_from_factors(factors):
    s = 1
    for p, c in factors.items():
        s *= (p ** (c + 1) - 1) // (p - 1)
    return s

def construct():
    start_time = time.time()
    DEADLINE = 140.0 # 10s margin
    
    # Fallback
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    # Step 1: Sieve to find candidate M
    # We can afford a larger sieve if we do it efficiently
    N = 5_000_000 # 5 million is fast
    # Let's try to go higher if time permits, but 5M is a safe bet for speed
    # Actually, let's do 10M if we can.
    
    try:
        import numpy as np
        N = 10_000_000
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        # Compute M = k * sigma(k)
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:N + 1]
        
        # Filter M <= 10^18
        mask = M_vals <= 10**18
        M_vals = M_vals[mask]
        k_vals = k_vals[mask]
        
        # Count occurrences
        # np.unique is O(N log N)
        unique_M, inverse = np.unique(M_vals, return_inverse=True)
        counts = np.bincount(inverse)
        
        # Get indices of M with highest counts
        # We want top candidates
        # Sort indices by count descending
        # But we only need top ~100
        top_indices = np.argsort(counts)[::-1][:100]
        
        candidates = []
        for idx in top_indices:
            if counts[idx] > best_count:
                M_cand = int(unique_M[idx])
                candidates.append((M_cand, int(counts[idx])))
                
    except Exception:
        candidates = []
        
    if time.time() - start_time > DEADLINE - 20:
        return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
        
    # Step 2: For each candidate, find ALL k
    for M_cand, initial_count in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        try:
            # Factor M
            factors_M = factorize(M_cand)
            
            # Generate all divisors of M
            divisors = get_divisors(factors_M)
            
            valid_ks = []
            sqrt_M = math.isqrt(M_cand)
            
            for k in divisors:
                if k > sqrt_M:
                    continue
                if k < 1:
                    continue
                
                # Check if k * sigma(k) == M
                # We need sigma(k).
                # We can compute sigma(k) from its factors.
                # But we don't have factors of k directly, only factors of M.
                # We can extract factors of k from factors_M.
                
                factors_k = {}
                temp_k = k
                for p, c in factors_M.items():
                    if temp_k % p == 0:
                        cnt = 0
                        while temp_k % p == 0:
                            temp_k //= p
                            cnt += 1
                        factors_k[p] = cnt
                if temp_k > 1:
                    factors_k[temp_k] = 1 # Should not happen if divisors are correct
                    
                sigma_k = sigma_from_factors(factors_k)
                
                if k * sigma_k == M_cand:
                    valid_ks.append(k)
                    
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M_cand
                best_k_list = valid_ks
                
        except Exception:
            continue
            
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
