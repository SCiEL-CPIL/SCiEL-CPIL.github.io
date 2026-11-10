"""Optimized program for finding M with many solutions to k*sigma(k) = M."""
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Phase 1: Sieve sigma up to N
    N = 8000000  # 8 million - finds more M values with multiple preimages
    
    # Compute sigma via sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    if time.time() > deadline - 60:
        # Emergency: return best found so far with minimal extension
        m_to_k = defaultdict(list)
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val <= 10**18:
                m_to_k[M_val].append(k)
        best_M, best_ks = _find_best_from_map(m_to_k, N)
        return _finalize(best_M, best_ks)
    
    # Group k by M_k = k * sigma(k)
    m_to_k = defaultdict(list)
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    # Find top candidates by count of k <= sqrt(M)
    candidates = []
    for M_val, ks in m_to_k.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) >= 2:
            candidates.append((len(valid_ks), M_val, valid_ks))
    
    candidates.sort(key=lambda x: -x[0])
    
    # Start with best candidate
    if candidates:
        best_count, best_M, best_ks = candidates[0]
    else:
        best_count, best_M, best_ks = 0, 333312, [336, 372, 434]
    
    # Phase 2: Extension search - find k > N such that k * sigma(k) = M
    # For each top candidate, check divisors k > N where k <= sqrt(M)
    # Use factorization to generate divisors efficiently
    
    # Precompute primes up to sqrt(10^18) = 10^9 for factorization
    # But that's too large. Instead, use trial division with wheel for factorization.
    
    top_candidates = candidates[:50]  # Check top 50 candidates
    
    for count, M_val, valid_ks in top_candidates:
        if time.time() > deadline - 15:
            break
        
        sqrt_M = math.isqrt(M_val)
        existing = set(valid_ks)
        ext_ks = list(valid_ks)
        
        # Generate divisors of M_val using factorization
        divisors = _get_divisors(M_val)
        
        # Check divisors k > N where k <= sqrt_M
        for k in sorted(divisors):
            if k > N and k <= sqrt_M and k not in existing:
                expected_sigma = M_val // k
                if _sigma_by_factorization(k) == expected_sigma:
                    ext_ks.append(k)
                    existing.add(k)
            
            if time.time() > deadline - 10:
                break
        
        if len(ext_ks) > best_count:
            best_M = M_val
            best_ks = sorted(ext_ks)
            best_count = len(ext_ks)
    
    # Phase 3: Also try M values with count=1 that might have extensions
    # (less likely but worth checking if time permits)
    if time.time() < deadline - 20:
        for count, M_val, valid_ks in candidates[50:200]:
            if time.time() > deadline - 10:
                break
            
            sqrt_M = math.isqrt(M_val)
            existing = set(valid_ks)
            ext_ks = list(valid_ks)
            
            divisors = _get_divisors(M_val)
            
            for k in sorted(divisors):
                if k > N and k <= sqrt_M and k not in existing:
                    expected_sigma = M_val // k
                    if _sigma_by_factorization(k) == expected_sigma:
                        ext_ks.append(k)
                        existing.add(k)
                
                if time.time() > deadline - 10:
                    break
            
            if len(ext_ks) > best_count:
                best_M = M_val
                best_ks = sorted(ext_ks)
                best_count = len(ext_ks)
    
    return _finalize(best_M, best_ks)

def _find_best_from_map(m_to_k, N):
    """Find best M from map with minimal extension."""
    best_count = 0
    best_M = 333312
    best_ks = [336, 372, 434]
    
    for M_val, ks in m_to_k.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_M = M_val
            best_ks = sorted(valid_ks)
            best_count = len(valid_ks)
    
    return best_M, best_ks

def _finalize(M, ks):
    """Ensure correct types and constraints."""
    M = int(M)
    ks = sorted(set(int(k) for k in ks))
    # Verify k <= isqrt(M)
    sqrt_M = math.isqrt(M)
    ks = [k for k in ks if k <= sqrt_M]
    return {'M': M, 'k': ks}

def _get_divisors(n):
    """Get all divisors of n using factorization."""
    if n <= 0:
        return []
    
    factors = _factorize(n)
    
    # Generate all divisors from prime factorization
    divisors = [1]
    for p, e in factors:
        current = list(divisors)
        p_power = 1
        for i in range(e):
            p_power *= p
            for d in current:
                divisors.append(d * p_power)
    
    return divisors

def _factorize(n):
    """Factorize n using trial division with wheel. Returns list of (prime, exponent)."""
    if n <= 1:
        return []
    
    factors = []
    temp = n
    
    # Divide by 2
    if temp % 2 == 0:
        count = 0
        while temp % 2 == 0:
            temp //= 2
            count += 1
        factors.append((2, count))
    
    # Divide by 3
    if temp % 3 == 0:
        count = 0
        while temp % 3 == 0:
            temp //= 3
            count += 1
        factors.append((3, count))
    
    # Wheel: 2, 3, 4, 6, 8, 12 pattern (skip multiples of 2 and 3)
    d = 5
    wheel = [2, 4, 2, 4, 2, 4, 6, 2, 6]  # increments to skip multiples of 2,3
    wheel_idx = 0
    
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors.append((d, count))
        
        d += wheel[wheel_idx]
        wheel_idx = (wheel_idx + 1) % len(wheel)
    
    if temp > 1:
        factors.append((temp, 1))
    
    return factors

def _sigma_by_factorization(n):
    """Compute sigma(n) by factoring n."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    factors = _factorize(n)
    result = 1
    for p, e in factors:
        # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
        p_power = p ** (e + 1)
        result *= (p_power - 1) // (p - 1)
    
    return result
