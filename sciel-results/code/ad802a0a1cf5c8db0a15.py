import time
import math
from collections import defaultdict

def _sigma_from_factors(factors):
    if not factors:
        return 1
    result = 1
    for p, e in factors.items():
        pe = pow(p, e + 1)
        result *= (pe - 1) // (p - 1)
    return result

def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _find_k_for_M(M, M_factors, deadline):
    """Find all k | M such that k*sigma(k) = M, using recursive search over prime power allocations."""
    results = []
    prime_list = sorted(M_factors.keys())
    exps = [M_factors[p] for p in prime_list]
    n_primes = len(prime_list)
    isqrt_M = math.isqrt(M)
    
    # Precompute sigma(p^e) for each prime and exponent
    sigma_table = {}
    for i, p in enumerate(prime_list):
        sigma_table[i] = [1]  # sigma(1) = 1
        s = 1
        pk = 1
        for e in range(1, exps[i] + 1):
            pk *= p
            s += pk
            sigma_table[i].append(s)
    
    def _rec(idx, k_val, sigma_val):
        if time.time() > deadline:
            return
        if idx == n_primes:
            if k_val * sigma_val == M and k_val <= isqrt_M and k_val >= 1:
                results.append(k_val)
            return
        
        p = prime_list[idx]
        max_e = exps[idx]
        
        for e in range(max_e + 1):
            new_k = k_val * (p ** e)
            if new_k > isqrt_M:
                break
            new_sigma = sigma_val * sigma_table[idx][e]
            
            # Pruning: new_k * new_sigma must be <= M
            if new_k * new_sigma > M:
                break
            # Pruning: M must be divisible by new_k
            if M % new_k != 0:
                continue
            # Pruning: remaining product must be achievable
            # The remaining sigma product must equal M / (new_k * new_sigma)
            # And the remaining k product must divide M / new_k
            remaining_needed = M // (new_k * new_sigma) if new_sigma > 0 else 0
            if remaining_needed < 1:
                break
            # Check that remaining_needed can be formed by sigma of remaining primes
            # Loose check: remaining_needed must be >= 1
            _rec(idx + 1, new_k, new_sigma)
    
    _rec(0, 1, 1)
    return results

def construct():
    deadline = time.time() + 135
    
    best_M = None
    best_ks = []
    
    # Generate candidate M values with many small prime factors
    # Strategy: take products of small prime powers
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    candidates = set()
    
    def _gen(idx, current):
        if current > 10**18:
            return
        if idx >= len(small_primes):
            if current > 1:
                candidates.add(current)
            return
        p = small_primes[idx]
        val = current
        for e in range(0, 61):
            if val > 10**18:
                break
            if val > 1:
                candidates.add(val)
            _gen(idx + 1, val)
            val *= p
    
    _gen(0, 1)
    
    # Score candidates: prefer those with many distinct small primes
    def _score(M):
        f = _factorize(M)
        distinct = len(f)
        total_exp = sum(f.values())
        # Prefer more distinct primes, fewer total exponents (so k can be small)
        return (distinct, -total_exp, M)
    
    scored = sorted(candidates, key=_score, reverse=True)
    
    # Try top candidates
    tried = 0
    for M in scored:
        if time.time() > deadline:
            break
        if tried > 3000:
            break
        
        M_factors = _factorize(M)
        ks = _find_k_for_M(M, M_factors, deadline)
        
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            if len(best_ks) >= 30:
                # Keep looking a bit more
                pass
        
        tried += 1
    
    # Verify best result
    if best_M is not None and best_ks:
        verified = []
        for k in best_ks:
            if time.time() > deadline:
                break
            factors = _factorize(k)
            sig = _sigma_from_factors(factors)
            if k * sig == best_M and k <= math.isqrt(best_M):
                verified.append(k)
        verified = sorted(set(verified))
        if verified:
            return {'M': int(best_M), 'k': [int(k) for k in verified[:1000]]}
    
    # Fallback: sieve approach
    m_to_ks = defaultdict(list)
    max_k = 100000
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        n = k
        sig = 1
        d = 2
        tmp = n
        while d * d <= tmp:
            if tmp % d == 0:
                pe = 1
                pk = 1
                while tmp % d == 0:
                    tmp //= d
                    pk *= d
                    pe += pk
                sig *= pe
            d += 1 if d == 2 else 2
        if tmp > 1:
            sig *= (1 + tmp)
        M = k * sig
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    for M, ks in m_to_ks.items():
        isqrt_M = math.isqrt(M)
        valid = [k for k in ks if k <= isqrt_M]
        if len(valid) > len(best_ks):
            best_M = M
            best_ks = valid
    
    if best_M is not None and best_ks:
        verified = sorted(set(best_ks))
        return {'M': int(best_M), 'k': [int(k) for k in verified[:1000]]}
    
    return {'M': 333312, 'k': [336, 372, 434]}
