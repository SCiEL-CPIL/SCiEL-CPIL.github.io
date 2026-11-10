"""Find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _isqrt(x):
    return math.isqrt(x)

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial division factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = (d ** (power + 1) - 1) // (d - 1)
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    if n <= 0:
        return []
    if n == 1:
        return []
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factors.append((d, power))
        d += 1
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _sigma_from_factors(factors):
    """Compute sigma(n) from its prime factorization [(p, e), ...]."""
    result = 1
    for p, e in factors:
        result *= (p ** (e + 1) - 1) // (p - 1)
    return result

def _dfs_prime_exponents(primes, max_exponent, max_k, start_idx=0, current_k=1, current_factors=None):
    """DFS over prime exponents to generate k values with controlled structure."""
    if current_factors is None:
        current_factors = []
    
    if current_k > max_k:
        return
    
    if start_idx == len(primes):
        yield current_k, list(current_factors)
        return
    
    p = primes[start_idx]
    # Try exponents 0 to max_exponent
    for e in range(max_exponent + 1):
        if current_k * (p ** e) > max_k:
            break
        current_factors.append((p, e))
        yield from _dfs_prime_exponents(primes, max_exponent, max_k, start_idx + 1, 
                                         current_k * (p ** e), current_factors)
        current_factors.pop()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Strategy: Use DFS over prime exponents to find structured k values
    # that produce many collisions in M = k * sigma(k)
    
    # Small primes to use in our structured search
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Generate candidate k values using DFS with limited exponents
    # We'll search for k where sigma(k)/k takes specific rational values
    # that lead to many collisions
    
    # Phase 1: Generate structured k values and group by M = k*sigma(k)
    m_to_k = defaultdict(list)
    
    # Try different maximum exponents to control the search space
    for max_exp in [1, 2, 3]:
        if time.time() > deadline - 20:
            break
        
        # Generate k values using DFS
        count = 0
        for k, factors in _dfs_prime_exponents(small_primes, max_exp, 10**12):
            count += 1
            if count % 100000 == 0 and time.time() > deadline - 10:
                break
            
            # Compute sigma(k) from factors
            sigma_k = _sigma_from_factors(factors)
            M = k * sigma_k
            
            if M <= 10**18:
                sqrt_M = _isqrt(M)
                if k <= sqrt_M:
                    m_to_k[M].append(k)
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Find best candidates from structured search
    if m_to_k:
        candidates = sorted(m_to_k.items(), key=lambda x: len(x[1]), reverse=True)
        top_candidates = candidates[:100]
        
        for M_val, ks in top_candidates:
            if time.time() > deadline - 5:
                break
            
            sqrt_M = _isqrt(M_val)
            valid_ks = sorted(set(k for k in ks if k <= sqrt_M))
            current_count = len(valid_ks)
            
            if current_count <= best_count:
                continue
            
            # Try to extend by finding more divisors k of M_val
            # that satisfy k * sigma(k) = M_val
            # Factor M_val to get all divisors
            m_factors = _factorize(M_val)
            
            # Generate all divisors from factorization
            def _gen_divisors(factors, idx=0, current=1):
                if idx == len(factors):
                    yield current
                    return
                p, e = factors[idx]
                for i in range(e + 1):
                    yield from _gen_divisors(factors, idx + 1, current * (p ** i))
            
            existing = set(valid_ks)
            for k in _gen_divisors(m_factors):
                if k <= 1 or k > sqrt_M or k in existing:
                    continue
                if _sigma_by_factorization(k) * k == M_val:
                    valid_ks.append(k)
                    existing.add(k)
            
            valid_ks = sorted(set(valid_ks))
            if len(valid_ks) > best_count:
                best_M = M_val
                best_ks = valid_ks
                best_count = len(valid_ks)
    
    # Phase 3: Also try a targeted search for known good structures
    # Known: M = 333312 has k = {336, 372, 434}
    # Try to find M with more solutions by targeting specific sigma/k ratios
    
    # Target ratios that are known to produce many solutions
    # sigma(k)/k = product of (p^(e+1)-1)/(p-1)/p^e
    # For k = 2^a * 3^b, sigma(k)/k = (2^(a+1)-1)/2^a * (3^(b+1)-1)/3^b
    
    # Try specific target M values that are highly composite
    # and check for multiple k solutions
    
    # Additional targeted search: look for M where multiple k work
    # by checking M values that are products of many small primes
    if time.time() < deadline - 5:
        # Try M = lcm(1..n) * small factor for various n
        for base in range(10, 30):
            if time.time() > deadline - 3:
                break
            # Compute lcm(1..base)
            lcm_val = 1
            for i in range(2, base + 1):
                lcm_val = lcm_val * i // math.gcd(lcm_val, i)
            
            for mult in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                M_test = lcm_val * mult
                if M_test > 10**18:
                    continue
                
                sqrt_M = _isqrt(M_test)
                ks_found = []
                
                # Check divisors k of M_test with k <= sqrt_M
                m_factors = _factorize(M_test)
                
                def _gen_divs(factors, idx=0, current=1):
                    if idx == len(factors):
                        yield current
                        return
                    p, e = factors[idx]
                    for i in range(e + 1):
                        yield from _gen_divs(factors, idx + 1, current * (p ** i))
                
                for k in _gen_divs(m_factors):
                    if k <= 1 or k > sqrt_M:
                        continue
                    if _sigma_by_factorization(k) * k == M_test:
                        ks_found.append(k)
                
                if len(ks_found) > best_count:
                    best_M = M_test
                    best_ks = sorted(ks_found)
                    best_count = len(ks_found)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # Final validation: verify all k satisfy k*sigma(k) == M and k <= sqrt(M)
    sqrt_M = _isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if k <= sqrt_M and _sigma_by_factorization(k) * k == best_M:
            final_ks.append(k)
    
    return {'M': best_M, 'k': final_ks}

# EVOLVE-BLOCK-END
