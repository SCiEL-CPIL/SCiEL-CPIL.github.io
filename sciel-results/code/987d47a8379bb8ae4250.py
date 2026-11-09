"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p**(exp + 1) - 1) // (p - 1)
        p += 1
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

def _factorize(n):
    """Factorize n into prime factors. Returns dict {prime: exponent}."""
    factors = {}
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors[p] = exp
        p += 1
    if temp > 1:
        factors[temp] = 1
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for i, (p, exp) in enumerate(zip(primes, exponents)):
        current = []
        for d in divisors:
            for e in range(exp + 1):
                current.append(d * (p ** e))
        divisors = current
    return divisors

def _find_k_for_M(M):
    """Find all k such that k * sigma(k) = M and k <= sqrt(M)."""
    if M < 1:
        return []
    
    max_k = math.isqrt(M)
    if max_k < 1:
        return []
    
    k_values = []
    
    # For each possible k from 1 to max_k, check if k * sigma(k) == M
    # But this is too slow for large M. Instead, use factorization approach.
    
    # Strategy: enumerate possible k by considering the structure
    # k must divide M, and sigma(k) = M/k
    # So k must be a divisor of M, and M/k must equal sigma(k)
    
    # Get all divisors of M that are <= max_k
    M_factors = _factorize(M)
    all_divs = _all_divisors_from_factors(M_factors)
    
    # Filter divisors <= max_k
    candidate_ks = [d for d in all_divs if d <= max_k]
    
    # For each candidate k, check if k * sigma(k) == M
    for k in candidate_ks:
        if k * _sigma(k) == M:
            k_values.append(k)
    
    return k_values

def _generate_candidate_Ms(max_primes=10, max_exp=8, deadline=None):
    """Generate candidate M values from products of small primes with varying exponents."""
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    candidates = set()
    
    # Generate M values as products of sigma(p^a) for small primes
    # sigma(p^a) = (p^(a+1) - 1) / (p - 1)
    # M = product of sigma(p_i^a_i) for various choices
    
    def _gen(idx, current_M, current_k):
        if deadline is not None and time.time() > deadline:
            return
        if idx >= len(small_primes):
            if 1 < current_M <= 10**18:
                candidates.add(current_M)
            return
        
        p = small_primes[idx]
        # Try exponent 0 (skip this prime) and exponents 1..max_exp
        for exp in range(0, max_exp + 1):
            if exp == 0:
                _gen(idx + 1, current_M, current_k)
            else:
                sigma_p = (p ** (exp + 1) - 1) // (p - 1)
                new_M = current_M * sigma_p
                if new_M > 10**18:
                    break
                new_k = current_k * (p ** exp)
                _gen(idx + 1, new_M, new_k)
    
    _gen(0, 1, 1)
    
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Generate candidate M values
    candidates = _generate_candidate_Ms(max_primes=8, max_exp=6, deadline=deadline)
    
    # Sort candidates by number of prime factors (heuristic: more factors = more solutions)
    def _num_prime_factors(n):
        f = _factorize(n)
        return sum(f.values())
    
    candidates.sort(key=lambda m: (-_num_prime_factors(m), m))
    
    # For each candidate M, find k values
    for M in candidates:
        if time.time() > deadline:
            break
        if M > 10**18:
            continue
        
        ks = _find_k_for_M(M)
        count = len(ks)
        
        if count > best_count:
            best_M = M
            best_ks = ks
            best_count = count
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}
# EVOLVE-BLOCK-END
