"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
from itertools import product

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n <= 1:
        return n
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            result *= (d**(exp+1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)
    return result

def _sigma_fast(n, primes, spf):
    """Compute sigma(n) using precomputed smallest prime factor."""
    if n <= 1:
        return n
    result = 1
    temp = n
    while temp > 1:
        p = spf[temp]
        exp = 0
        power = 1
        while temp % p == 0:
            temp //= p
            exp += 1
            power *= p
        result *= (power * p - 1) // (p - 1)
    return result

def _build_spf(limit):
    """Build smallest prime factor array up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _find_ks_for_M(M, spf, deadline):
    """Find all k such that k * sigma(k) = M and k <= isqrt(M)."""
    sqrt_M = isqrt(M)
    if sqrt_M < 1:
        return []
    
    valid_ks = []
    for k in range(1, sqrt_M + 1):
        if k * _sigma_fast(k, None, spf) == M:
            valid_ks.append(k)
        if time.time() > deadline:
            break
    return valid_ks

def _generate_candidate_Ms(deadline, max_candidates=500):
    """Generate candidate M values likely to have many solutions.
    
    Strategy: Generate M values from products of small primes with various exponents,
    focusing on highly composite numbers that are likely to have many divisors k
    satisfying k * sigma(k) = M.
    """
    candidates = set()
    
    # Base primes to use
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Generate M values by combining small primes with exponents
    # Focus on numbers with many divisors
    max_exp = 8
    
    # Generate combinations of prime powers
    prime_powers = {}
    for p in small_primes:
        powers = []
        val = 1
        for e in range(1, max_exp + 1):
            val *= p
            if val > 10**15:
                break
            powers.append(val)
        prime_powers[p] = powers
    
    # Generate M values as products of prime powers
    # Use a limited search space to stay within time budget
    def gen_products(idx, current_M, current_count):
        if time.time() > deadline:
            return
        if idx == len(small_primes):
            if 1 <= current_M <= 10**18:
                candidates.add(current_M)
            return
        
        p = small_primes[idx]
        # Include option of not using this prime (exponent 0)
        gen_products(idx + 1, current_M, current_count + 1)
        
        for power in prime_powers[p]:
            new_M = current_M * power
            if new_M > 10**18:
                break
            gen_products(idx + 1, new_M, current_count + 1)
    
    gen_products(0, 1, 0)
    
    # Also add some known good M values
    candidates.add(333312)
    
    # Sort by number of divisors (more divisors = more potential k values)
    # But computing divisors is expensive, so just return a subset
    return list(candidates)[:max_candidates]

def _count_divisors(n, spf):
    """Count divisors of n using smallest prime factor."""
    if n <= 1:
        return 1
    count = 1
    temp = n
    while temp > 1:
        p = spf[temp]
        exp = 0
        while temp % p == 0:
            temp //= p
            exp += 1
        count *= (exp + 1)
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Known good answer
    M0, k0 = 333312, [336, 372, 434]
    best_M, best_k = M0, k0
    
    # Build SPF table for fast sigma computation
    # We need spf up to isqrt(10^18) = 10^9, but that's too large
    # Instead, use trial division for sigma, but cache results
    
    sigma_cache = {}
    
    def _sigma_cached(n):
        if n in sigma_cache:
            return sigma_cache[n]
        result = _sigma(n)
        sigma_cache[n] = result
        return result
    
    # Generate candidate M values
    candidates = _generate_candidate_Ms(deadline, max_candidates=200)
    
    # Sort candidates by number of divisors (proxy for potential solutions)
    # Use a simple heuristic: prefer numbers with many small prime factors
    def divisor_heuristic(M):
        # Count total prime factors with multiplicity as a proxy
        temp = M
        count = 0
        d = 2
        while d * d <= temp and d < 100:
            while temp % d == 0:
                temp //= d
                count += 1
            d += 1
        if temp > 1:
            count += 1
        return count
    
    candidates.sort(key=divisor_heuristic, reverse=True)
    
    # For each candidate M, find all valid k
    for M in candidates:
        if time.time() > deadline:
            break
        
        sqrt_M = isqrt(M)
        if sqrt_M < 1 or sqrt_M > 500000:  # Limit search space
            continue
        
        valid_ks = []
        for k in range(1, sqrt_M + 1):
            if k * _sigma_cached(k) == M:
                valid_ks.append(k)
            if time.time() > deadline:
                break
        
        if len(valid_ks) > len(best_k):
            best_M = M
            best_k = valid_ks
    
    # Also try direct enumeration of k up to a limit as a fallback
    if time.time() < deadline - 10:
        max_k = 200000
        m_to_ks = defaultdict(list)
        for k in range(1, max_k + 1):
            M_val = k * _sigma_cached(k)
            if M_val <= 10**18:
                m_to_ks[M_val].append(k)
            if time.time() > deadline - 5:
                break
        
        for M_val, ks in m_to_ks.items():
            if len(ks) <= len(best_k):
                continue
            sqrt_M = isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            if len(valid_ks) > len(best_k):
                best_M = M_val
                best_k = valid_ks
    
    # Final verification
    valid_k = []
    for k in best_k:
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= isqrt(best_M):
            if k * _sigma_cached(k) == best_M:
                valid_k.append(k)
    
    if not valid_k:
        best_M, best_k = M0, k0
        valid_k = [k for k in best_k if k * _sigma_cached(k) == best_M and k <= isqrt(best_M)]
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in valid_k]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
