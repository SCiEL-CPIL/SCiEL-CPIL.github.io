"""Find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _isqrt(x):
    return math.isqrt(x)

def _factorize(n):
    """Return list of (prime, exponent) pairs for n using trial division."""
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

def _gen_divisors(factors, idx=0, current=1):
    """Generate all divisors from prime factorization."""
    if idx == len(factors):
        yield current
        return
    p, e = factors[idx]
    for i in range(e + 1):
        yield from _gen_divisors(factors, idx + 1, current * (p ** i))

def _gen_divisors_limited(factors, idx=0, current=1, limit=None):
    """Generate all divisors from prime factorization, skipping those > limit."""
    if idx == len(factors):
        if limit is None or current <= limit:
            yield current
        return
    p, e = factors[idx]
    for i in range(e + 1):
        new_current = current * (p ** i)
        if limit is not None and new_current > limit:
            break
        yield from _gen_divisors_limited(factors, idx + 1, new_current, limit)

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
    for e in range(max_exponent + 1):
        if current_k * (p ** e) > max_k:
            break
        current_factors.append((p, e))
        yield from _dfs_prime_exponents(primes, max_exponent, max_k, start_idx + 1, 
                                         current_k * (p ** e), current_factors)
        current_factors.pop()

def _search_structured(primes, max_exp, max_k, deadline):
    """Generate structured k values and group by M = k*sigma(k)."""
    m_to_k = defaultdict(list)
    count = 0
    for k, factors in _dfs_prime_exponents(primes, max_exp, max_k):
        count += 1
        if count % 50000 == 0 and time.time() > deadline - 15:
            break
        
        sigma_k = _sigma_from_factors(factors)
        M = k * sigma_k
        
        if M <= 10**18:
            sqrt_M = _isqrt(M)
            if k <= sqrt_M:
                m_to_k[M].append(k)
    
    return m_to_k

def _extend_with_divisors(M_val, existing_ks, deadline):
    """Try to find more k values that satisfy k*sigma(k) = M_val."""
    sqrt_M = _isqrt(M_val)
    valid_ks = list(set(k for k in existing_ks if k <= sqrt_M))
    existing = set(valid_ks)
    
    m_factors = _factorize(M_val)
    
    for k in _gen_divisors_limited(m_factors, limit=sqrt_M):
        if k <= 1 or k in existing:
            continue
        if _sigma_by_factorization(k) * k == M_val:
            valid_ks.append(k)
            existing.add(k)
    
    return sorted(valid_ks)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Phase 1: Structured search over small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    
    for max_exp in [1, 2, 3]:
        if time.time() > deadline - 30:
            break
        
        m_to_k = _search_structured(small_primes, max_exp, 10**12, deadline)
        
        if m_to_k:
            candidates = sorted(m_to_k.items(), key=lambda x: len(x[1]), reverse=True)
            top_candidates = candidates[:50]
            
            for M_val, ks in top_candidates:
                if time.time() > deadline - 10:
                    break
                
                valid_ks = _extend_with_divisors(M_val, ks, deadline)
                current_count = len(valid_ks)
                
                if current_count > best_count:
                    best_M = M_val
                    best_ks = valid_ks
                    best_count = current_count
    
    # Phase 2: Targeted search for highly composite M values
    if time.time() < deadline - 10:
        # Try M = lcm(1..n) * small factor
        for base in range(8, 25):
            if time.time() > deadline - 5:
                break
            lcm_val = 1
            for i in range(2, base + 1):
                lcm_val = lcm_val * i // math.gcd(lcm_val, i)
            
            for mult in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                if time.time() > deadline - 3:
                    break
                M_test = lcm_val * mult
                if M_test > 10**18:
                    continue
                
                valid_ks = _extend_with_divisors(M_test, [], deadline)
                
                if len(valid_ks) > best_count:
                    best_M = M_test
                    best_ks = valid_ks
                    best_count = len(valid_ks)
    
    # Phase 3: Search for M values that are products of many small prime powers
    if time.time() < deadline - 8:
        # Generate candidate M values as products of small prime powers
        candidate_Ms = set()
        
        # Products of powers of 2, 3, 5, 7
        for a in range(0, 60):
            for b in range(0, 38):
                for c in range(0, 26):
                    for d in range(0, 20):
                        M_val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if M_val <= 10**18 and M_val > 0:
                            candidate_Ms.add(M_val)
        
        # Check a subset of these candidates
        candidate_list = list(candidate_Ms)
        # Sort by number of prime factors (heuristic: more factors may mean more solutions)
        candidate_list.sort(key=lambda x: len(_factorize(x)), reverse=True)
        
        for M_val in candidate_list[:200]:
            if time.time() > deadline - 5:
                break
            
            valid_ks = _extend_with_divisors(M_val, [], deadline)
            
            if len(valid_ks) > best_count:
                best_M = M_val
                best_ks = valid_ks
                best_count = len(valid_ks)
    
    # Phase 4: Final targeted search around known good structures
    if time.time() < deadline - 5:
        # Try M values near the current best
        for offset in range(-1000, 1001, 100):
            if time.time() > deadline - 3:
                break
            M_test = best_M + offset
            if M_test <= 0 or M_test > 10**18:
                continue
            
            valid_ks = _extend_with_divisors(M_test, [], deadline)
            
            if len(valid_ks) > best_count:
                best_M = M_test
                best_ks = valid_ks
                best_count = len(valid_ks)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # Final validation: verify all k satisfy k*sigma(k) == M and k <= sqrt(M)
    sqrt_M = _isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if 1 <= k <= sqrt_M and _sigma_by_factorization(k) * k == best_M:
            final_ks.append(k)
    
    return {'M': best_M, 'k': final_ks}

# EVOLVE-BLOCK-END
