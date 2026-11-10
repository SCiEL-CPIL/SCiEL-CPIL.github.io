"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

# Cache for tau_perp results
_tau_perp_cache = {}

def _factorize(n):
    """Factorize n into prime factors with exponents."""
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
        p += 1 if p == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _generate_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_power = 1
        for _ in range(e):
            p_power *= p
            for d in divisors:
                new_divs.append(d * p_power)
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n in _tau_perp_cache:
        return _tau_perp_cache[n]
    
    factors = _factorize(n)
    divisors = _generate_divisors_from_factors(factors)
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    
    _tau_perp_cache[n] = count
    return count

def _try_exponent_combinations(primes, max_exponent, deadline, best_n, best_score):
    """Systematically try exponent combinations for given primes."""
    n = len(primes)
    if n == 0:
        return best_n, best_score
    
    # Recursive search with pruning
    def search(idx, current_product, exponents):
        nonlocal best_n, best_score
        if time.time() > deadline - 5:
            return
        
        if idx == n:
            if current_product >= 2 and current_product <= 10**18:
                score = _tau_perp(current_product)
                if score > best_score:
                    best_score = score
                    best_n = current_product
            return
        
        # Try all possible exponents for prime idx
        p = primes[idx]
        for exp in range(1, max_exponent + 1):
            new_product = current_product * (p ** exp)
            if new_product > 10**18:
                break
            exponents.append(exp)
            search(idx + 1, new_product, exponents)
            exponents.pop()
    
    search(0, 1, [])
    return best_n, best_score

def construct():
    """Construct an integer n <= 10^18 maximizing tau_perp(n)."""
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (5s buffer)
    
    # Base primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 1
    best_score = 0
    
    # Phase 1: Try primorials (all exponents = 1)
    product = 1
    for k in range(1, len(base_primes) + 1):
        product *= base_primes[k - 1]
        if product > 10**18:
            break
        score = _tau_perp(product)
        if score > best_score:
            best_score = score
            best_n = product
    
    # Phase 2: Systematic search with limited exponents for smaller prime sets
    # Start with fewer primes but allow higher exponents
    for num_primes in range(8, 16):
        if time.time() > deadline - 20:
            break
        
        selected_primes = base_primes[:num_primes]
        
        # Determine max exponent based on prime size
        # For small primes, we can afford higher exponents
        if num_primes <= 10:
            max_exp = 4
        elif num_primes <= 12:
            max_exp = 3
        else:
            max_exp = 2
        
        best_n, best_score = _try_exponent_combinations(
            selected_primes, max_exp, deadline, best_n, best_score
        )
    
    # Phase 3: Local search - try single exponent increases from current best
    if best_n > 1:
        factors = _factorize(best_n)
        factor_dict = dict(factors)
        
        # Try increasing each exponent by 1
        for p, e in factors:
            if time.time() > deadline - 10:
                break
            test_n = best_n * p
            if test_n <= 10**18:
                score = _tau_perp(test_n)
                if score > best_score:
                    best_score = score
                    best_n = test_n
        
        # Try decreasing each exponent by 1 (if > 1) and see if we can add a new prime
        for p, e in factors:
            if time.time() > deadline - 10:
                break
            if e > 1:
                test_n = best_n // p
                # Try adding a new prime that wasn't in the factorization
                for new_p in base_primes:
                    if new_p not in factor_dict:
                        candidate = test_n * new_p
                        if candidate <= 10**18 and candidate >= 2:
                            score = _tau_perp(candidate)
                            if score > best_score:
                                best_score = score
                                best_n = candidate
                                break
    
    return best_n
# EVOLVE-BLOCK-END
