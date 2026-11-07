import time
import math
from collections import defaultdict
from array import array

def _sigma_from_factorization(factors):
    """Compute sigma(n) from prime factorization [(p, e), ...]"""
    result = 1
    for p, e in factors:
        s = 1
        pe = 1
        for _ in range(e):
            pe *= p
            s += pe
        result *= s
    return result

def _factorize_small(n):
    """Factorize n using trial division for small n, return list of (p, e)"""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _find_k_for_m(M, k_max):
    """Find all k <= k_max such that k * sigma(k) = M"""
    ks = []
    # k must divide M
    # Iterate over divisors of M up to k_max
    # First, get factorization of M
    factors = _factorize_small(M)
    
    # Generate all divisors of M
    divisors = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for exp in range(e + 1):
            for d in divisors:
                new_divs.append(d * pe)
            pe *= p
        divisors = new_divs
    
    for k in divisors:
        if k > k_max:
            continue
        # Compute sigma(k)
        kfactors = _factorize_small(k)
        sk = _sigma_from_factorization(kfactors)
        if k * sk == M:
            ks.append(k)
    
    return ks

def _search_rich_m(deadline):
    """Search for M with high collision density by examining M with many divisors"""
    start = time.time()
    best_result = {'M': 336, 'k': [12, 14]}
    best_count = 2
    
    # Strategy: Look at M that are products of small primes
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * ...
    # We want M to have many divisors k where sigma(k) = M/k
    
    # Start with M values that are highly composite
    # These tend to have many divisors, increasing chance of collisions
    
    # Precompute small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate candidate M values from products of small prime powers
    # Focus on M up to 10^18
    max_m = 10**18
    
    candidates = set()
    
    # Add some known highly composite numbers and their multiples
    # Start with products of first few primes
    def gen_candidates(prime_list, max_val, current=1, idx=0):
        if time.time() > deadline - 5:
            return
        if current > max_val:
            return
        if idx == len(prime_list):
            candidates.add(current)
            return
        p = prime_list[idx]
        val = current
        while val <= max_val:
            gen_candidates(prime_list, max_val, val, idx + 1)
            if val > max_val // p:
                break
            val *= p
    
    # Generate candidates from first 8 primes (2,3,5,7,11,13,17,19)
    # This gives us M with many divisors
    gen_candidates(small_primes[:8], max_m)
    
    # Also add some targeted candidates: M = k * sigma(k) for small k
    # where k has a specific structure
    for k in range(2, 100000):
        if time.time() > deadline - 5:
            break
        # Quick check: compute sigma(k) approximately
        # Only consider k with small prime factors
        kfactors = _factorize_small(k)
        if kfactors and all(p <= 100 for p, e in kfactors):
            sk = _sigma_from_factorization(kfactors)
            m = k * sk
            if m <= max_m:
                candidates.add(m)
    
    if time.time() > deadline - 10:
        return best_result
    
    # For each candidate M, find all k such that k * sigma(k) = M
    # and k <= sqrt(M)
    for M in candidates:
        if time.time() > deadline - 10:
            break
        if M < 1 or M > max_m:
            continue
        sqrt_m = math.isqrt(M)
        if sqrt_m < 1:
            continue
        
        ks = _find_k_for_m(M, sqrt_m)
        
        if len(ks) > best_count:
            best_count = len(ks)
            best_result = {'M': M, 'k': ks}
            
        # Early termination if we find a very good result
        if best_count >= 10:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Fallback result
    fallback = {'M': 336, 'k': [12, 14]}
    
    # First, try the targeted search for rich M values
    # This is faster and may find better results
    try:
        targeted_result = _search_rich_m(deadline - 30)
        if targeted_result['k']:
            # Validate
            M = targeted_result['M']
            ks = targeted_result['k']
            sqrt_m = math.isqrt(M)
            valid_ks = [k for k in ks if 1 <= k <= sqrt_m]
            if valid_ks:
                targeted_result['k'] = valid_ks
                return {'M': int(M), 'k': [int(k) for k in valid_ks]}
    except Exception:
        pass
    
    # If targeted search didn't find good results, use the sieve approach
    N = 20_000_000
    
    try:
        # Allocate arrays
        sigma = array('Q', [0]) * (N + 1)
        spp = array('Q', [0]) * (N + 1)
        p_power = array('Q', [0]) * (N + 1)
        lp = array('I', [0]) * (N + 1)
        
        sigma[1] = 1
        spp[1] = 1
        p_power[1] = 1
        
        primes = []
        
        # Linear sieve
        for i in range(2, N + 1):
            if lp[i] == 0:
                lp[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                spp[i] = 1 + i
                p_power[i] = i
                
            for p in primes:
                if p > lp[i] or i * p > N:
                    break
                ip = i * p
                lp[ip] = p
                if i % p == 0:
                    p_power[ip] = p_power[i] * p
                    spp[ip] = spp[i] + p_power[ip]
                    sigma[ip] = sigma[i] // spp[i] * spp[ip]
                    break
                else:
                    p_power[ip] = p
                    spp[ip] = 1 + p
                    sigma[ip] = sigma[i] * (1 + p)
                    
        if time.time() > deadline - 20:
            return fallback
            
        freq = defaultdict(int)
        max_m = 10**18
        
        for k in range(1, N + 1):
            m_val = k * sigma[k]
            if m_val <= max_m:
                freq[m_val] += 1
                
        if not freq:
            return fallback
            
        best_count = 0
        best_m = 1
        for m, c in freq.items():
            if c > best_count:
                best_count = c
                best_m = m
                
        # Collect k's for best_m
        best_ks = []
        for k in range(1, N + 1):
            if k * sigma[k] == best_m:
                best_ks.append(k)
                
        if not best_ks:
            return fallback
            
        # Validate k <= sqrt(M)
        sqrt_m = math.isqrt(best_m)
        valid_ks = [k for k in best_ks if k <= sqrt_m]
        
        if not valid_ks:
            return fallback
            
        return {'M': int(best_m), 'k': [int(k) for k in valid_ks]}
        
    except Exception:
        return fallback

if __name__ == "__main__":
    result = construct()
    print(result)
