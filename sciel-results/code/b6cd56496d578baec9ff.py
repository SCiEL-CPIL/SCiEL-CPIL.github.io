"""Rewritten program: find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

def _isqrt(x):
    return math.isqrt(x)

def _sigma_by_factorization(n):
    """Compute sigma(n) by factoring n."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = 1
            d_power = 1
            for _ in range(power):
                d_power *= d
                factor += d_power
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _linear_sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a linear sieve."""
    if n < 1:
        return []
    sigma = [0] * (n + 1)
    sigma[1] = 1
    is_prime = [True] * (n + 1)
    primes = []
    # sigma_p[i] = sigma of the highest prime power of i (i.e., sigma(p^a) where p^a || i)
    sigma_p = [0] * (n + 1)
    sigma_p[1] = 1
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            sigma[i] = 1 + i
            sigma_p[i] = 1 + i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                # p divides i, so p^a || i for some a >= 1
                # i * p has p^(a+1) as highest prime power
                sigma_p[i * p] = sigma_p[i] + i * p
                sigma[i * p] = sigma[i] // sigma_p[i] * sigma_p[i * p]
                break
            else:
                # p does not divide i
                sigma_p[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    return sigma

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Strategy:
    # 1) Sieve sigma up to N using linear sieve (fast)
    # 2) For k in 1..N, compute M_k = k*sigma(k), group by M
    # 3) Find M with most k's (all k <= sqrt(M) automatically)
    # 4) Extend: for the best M, find additional k > N that divide M and satisfy sigma(k) = M/k
    
    # Choose N based on available time
    # Linear sieve up to 10M takes ~1-2s, up to 50M takes ~5-10s
    # Let's start with a reasonable N and see how much time we have
    
    N = 10000000  # 10 million
    
    sigma = _linear_sigma_sieve(N)
    
    # Group k by M_k = k * sigma(k)
    m_to_k = defaultdict(list)
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    # Find the M with the most solutions
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M_val, ks in m_to_k.items():
        if len(ks) > best_count:
            # Verify all k <= sqrt(M_val)
            sqrt_M = _isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            if len(valid_ks) > best_count:
                best_M = M_val
                best_ks = sorted(valid_ks)
                best_count = len(valid_ks)
    
    # Now try to extend best_M: search for k > N such that k * sigma(k) = best_M
    # sigma(k) = best_M / k, so k must divide best_M
    M_best = best_M
    target = M_best
    
    # Find divisors of target that are > N
    # k must satisfy: k | target and sigma(k) = target // k
    # Also k <= sqrt(target)
    max_k = _isqrt(target)
    
    # Get all divisors of target
    divisors = set()
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
        i += 1
    
    # Check divisors k > N (and not already in best_ks) where k <= max_k
    existing_ks = set(best_ks)
    extended_ks = list(best_ks)
    
    for k in sorted(divisors):
        if k > N and k <= max_k and k not in existing_ks:
            # Check if sigma(k) == target // k
            expected_sigma = target // k
            # Compute sigma(k) by factoring
            if _sigma_by_factorization(k) == expected_sigma:
                extended_ks.append(k)
                existing_ks.add(k)
        
        if time.time() > deadline - 20:
            break
    
    if len(extended_ks) > best_count:
        best_ks = extended_ks
        best_count = len(extended_ks)
    
    # If we still have time, try a larger sieve to find better M
    elapsed = time.time() - (deadline - 140)
    if elapsed < 100:  # Still have 40+ seconds
        # Try larger N
        N2 = 30000000  # 30 million
        try:
            sigma2 = _linear_sigma_sieve(N2)
            m_to_k2 = defaultdict(list)
            for k in range(1, N2 + 1):
                M_val = k * sigma2[k]
                if M_val <= 10**18:
                    m_to_k2[M_val].append(k)
            
            for M_val, ks in m_to_k2.items():
                if len(ks) > best_count:
                    sqrt_M = _isqrt(M_val)
                    valid_ks = [k for k in ks if k <= sqrt_M]
                    if len(valid_ks) > best_count:
                        best_M = M_val
                        best_ks = sorted(valid_ks)
                        best_count = len(valid_ks)
            
            # Extend again
            target = best_M
            max_k = _isqrt(target)
            divisors = set()
            i = 1
            while i * i <= target:
                if target % i == 0:
                    divisors.add(i)
                    divisors.add(target // i)
                i += 1
            
            existing_ks = set(best_ks)
            extended_ks = list(best_ks)
            
            for k in sorted(divisors):
                if k > N2 and k <= max_k and k not in existing_ks:
                    expected_sigma = target // k
                    if _sigma_by_factorization(k) == expected_sigma:
                        extended_ks.append(k)
                        existing_ks.add(k)
                
                if time.time() > deadline - 15:
                    break
            
            if len(extended_ks) > best_count:
                best_ks = extended_ks
                best_count = len(extended_ks)
        except MemoryError:
            pass
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
