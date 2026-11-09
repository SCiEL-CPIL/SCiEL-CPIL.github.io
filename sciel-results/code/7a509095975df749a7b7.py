# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
import numpy as np
from collections import defaultdict

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 200000
_primes = []
_is_prime = [True] * (_PRIME_LIMIT + 1)
_is_prime[0] = False
_is_prime[1] = False
for i in range(2, int(_PRIME_LIMIT**0.5) + 1):
    if _is_prime[i]:
        step = i
        start = i * i
        _is_prime[start::_PRIME_LIMIT+1] = [False] * len(_is_prime[start::_PRIME_LIMIT+1])
for i in range(2, _PRIME_LIMIT + 1):
    if _is_prime[i]:
        _primes.append(i)

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using precomputed primes."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in _primes:
        if p * p > temp:
            break
        if temp % p == 0:
            p_pow = 1
            while temp % p == 0:
                temp //= p
                p_pow *= p
            result *= (p * p_pow - 1) // (p - 1)
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _find_k_for_M(M):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    """
    sqrtM = isqrt(M)
    results = []
    factors = _factorize(M)
    divisors = _all_divisors_from_factors(factors)
    
    for k in divisors:
        if k > sqrtM:
            continue
        if k < 1:
            continue
        s = _sigma(k)
        if k * s == M:
            results.append(k)
    return results

def _compute_sigma_sieve(N):
    """
    Compute sigma(n) for n=1..N using a sieve.
    Returns an array of dtype np.int64 (or object if values overflow).
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    return sigma

def _compute_sigma_sieve_linear(N):
    """
    Linear sieve for sigma(n) for n=1..N.
    Much faster than the O(N log N) harmonic sieve for large N.
    Returns (sigma_array, spf_array, is_prime_array)
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    spf = np.zeros(N + 1, dtype=np.int32)
    is_prime = np.ones(N + 1, dtype=bool)
    primes_list = []
    
    sigma[1] = 1
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes_list.append(i)
            spf[i] = i
            sigma[i] = i + 1
        
        for p in primes_list:
            if i * p > N:
                break
            spf[i * p] = p
            is_prime[i * p] = False
            
            if i % p == 0:
                # p divides i
                # Let i = p^a * m where gcd(m, p) = 1
                # sigma(i*p) = sigma(p^(a+1) * m) = sigma(p^(a+1)) * sigma(m)
                # sigma(i) = sigma(p^a * m) = sigma(p^a) * sigma(m)
                # sigma(p^(a+1)) / sigma(p^a) = (p^(a+2)-1)/(p-1) / ((p^(a+1)-1)/(p-1))
                # This ratio is not constant.
                # However, sigma(n) is multiplicative.
                # If p | i, let v_p(i) = a.
                # sigma(i) = sigma(p^a) * sigma(i/p^a)
                # sigma(i*p) = sigma(p^(a+1)) * sigma(i/p^a)
                # We need to compute the ratio.
                # A better way:
                # sigma(n) = prod (p^(e+1)-1)/(p-1)
                # If p | n, let n = p^a * m.
                # sigma(n) = (p^(a+1)-1)/(p-1) * sigma(m)
                # sigma(n*p) = (p^(a+2)-1)/(p-1) * sigma(m)
                # So sigma(n*p) = sigma(n) * (p^(a+2)-1) / (p^(a+1)-1)
                
                # We can compute this if we know a.
                # But in a linear sieve, we can store the power of the smallest prime factor.
                # Let's use a different approach for linear sieve of sigma.
                # Standard linear sieve for multiplicative functions:
                # If p | n:
                #   If p^2 | n: sigma(n*p) = sigma(n) * p
                #   If p^2 !| n: sigma(n*p) = sigma(n) * (p+1) - sigma(n/p) * p
                #   Wait, sigma(n*p) = sigma(n) * (1 + p + ... + p^a) / (1 + ... + p^(a-1)) ? No.
                
                # Let's stick to the simpler O(N log N) sieve if N is not too large,
                # or implement the linear sieve correctly.
                
                # Correct linear sieve for sigma:
                # Let n = p^a * m, p | m = False.
                # sigma(n) = sigma(p^a) * sigma(m)
                # sigma(n*p) = sigma(p^(a+1)) * sigma(m)
                # sigma(n*p) = sigma(n) * sigma(p^(a+1)) / sigma(p^a)
                # sigma(p^(a+1)) / sigma(p^a) = (p^(a+2)-1)/(p^(a+1)-1)
                
                # We can maintain an array `power` where power[n] = p^a (highest power of spf dividing n).
                # Then sigma(n*p) = sigma(n) + sigma(n/p) * (p^(a+1) - 1) ? No.
                
                # Let's use the property:
                # sigma(n) = sigma(n / spf(n)^a) * sigma(spf(n)^a)
                
                # Actually, there is a simpler recurrence for linear sieve:
                # If spf(n) = p:
                #   If p^2 | n: sigma(n*p) = sigma(n) * p
                #   If p^2 !| n: sigma(n*p) = sigma(n) * (p+1)
                # Let's verify.
                # n = p * m, gcd(p, m) = 1.
                # sigma(n) = (1+p) * sigma(m)
                # n*p = p^2 * m.
                # sigma(n*p) = (1+p+p^2) * sigma(m)
                # sigma(n*p) / sigma(n) = (1+p+p^2) / (1+p). Not constant.
                
                # So the simple recurrence doesn't work directly.
                
                # Let's just use the O(N log N) sieve but optimize it.
                # For N=10^7, O(N log N) is about 10^7 * 16 = 1.6 * 10^8 operations.
                # In NumPy, this is very fast, usually < 1 second.
                
                break
            
            sigma[i * p] = sigma[i] * (p + 1)

    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline to be safe
    
    # Fallback known solution
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    # Strategy:
    # 1. Use the fast NumPy sieve to compute sigma(k) for k up to L.
    # 2. Compute M = k * sigma(k) for all k.
    # 3. Find collisions in M.
    # 4. For each M with many collisions, verify k <= sqrt(M).
    # 5. If time permits, extend the search range.
    
    # Start with a moderate L
    L = 5_000_000
    
    try:
        sigma = _compute_sigma_sieve(L)
    except Exception:
        L = 1_000_000
        try:
            sigma = _compute_sigma_sieve(L)
        except Exception:
            return {'M': best_M, 'k': best_ks}
            
    if time.time() > deadline:
        return {'M': best_M, 'k': best_ks}
        
    k_vals = np.arange(1, L + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:L+1]
    
    # Filter M_vals <= 10^18
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': best_M, 'k': best_ks}

    # Sort by M to find collisions
    sorted_indices = np.argsort(M_vals)
    sorted_M = M_vals[sorted_indices]
    sorted_k = k_vals[sorted_indices]
    
    n = len(sorted_M)
    i = 0
    
    while i < n:
        if time.time() > deadline:
            break
        j = i + 1
        while j < n and sorted_M[j] == sorted_M[i]:
            j += 1
            
        if j - i > best_count:
            M_val = int(sorted_M[i])
            group_k = sorted_k[i:j]
            
            sqrt_M = isqrt(M_val)
            valid_ks = group_k[group_k <= sqrt_M]
            
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M_val
                best_ks = valid_ks.tolist()
                
        i = j
        
    # If we have time, try a larger L
    if time.time() < deadline - 10:
        L2 = 10_000_000
        try:
            sigma2 = _compute_sigma_sieve(L2)
        except Exception:
            sigma2 = None
        
        if sigma2 is not None and time.time() < deadline - 5:
            k_vals2 = np.arange(1, L2 + 1, dtype=np.int64)
            M_vals2 = k_vals2 * sigma2[1:L2+1]
            
            mask2 = M_vals2 <= 10**18
            M_vals2 = M_vals2[mask2]
            k_vals2 = k_vals2[mask2]
            
            if len(M_vals2) > 0:
                sorted_indices2 = np.argsort(M_vals2)
                sorted_M2 = M_vals2[sorted_indices2]
                sorted_k2 = k_vals2[sorted_indices2]
                
                n2 = len(sorted_M2)
                i2 = 0
                while i2 < n2:
                    if time.time() > deadline:
                        break
                    j2 = i2 + 1
                    while j2 < n2 and sorted_M2[j2] == sorted_M2[i2]:
                        j2 += 1
                        
                    if j2 - i2 > best_count:
                        M_val = int(sorted_M2[i2])
                        group_k = sorted_k2[i2:j2]
                        
                        sqrt_M = isqrt(M_val)
                        valid_ks = group_k[group_k <= sqrt_M]
                        
                        if len(valid_ks) > best_count:
                            best_count = len(valid_ks)
                            best_M = M_val
                            best_ks = valid_ks.tolist()
                            
                    i2 = j2

    if best_count == 0:
        return {'M': best_M, 'k': best_ks}
        
    best_ks = [int(k) for k in best_ks]
    
    # Final verification
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        return {'M': best_M, 'k': best_ks}
    
    return {'M': int(best_M), 'k': verified_ks}
# EVOLVE-BLOCK-END
