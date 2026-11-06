import time
import math
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    factors = {}
    tmp = n
    d = 2
    while d * d <= tmp:
        while tmp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            tmp //= d
        d += 1
    if tmp > 1:
        factors[tmp] = factors.get(tmp, 0) + 1
    
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    return divs

def _score(n):
    """Compute tau_perp(n): number of adjacent divisor pairs with gcd 1."""
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _factorize(n):
    """Return list of (prime, exponent) pairs."""
    factors = []
    tmp = n
    d = 2
    while d * d <= tmp:
        if tmp % d == 0:
            e = 0
            while tmp % d == 0:
                tmp //= d
                e += 1
            factors.append((d, e))
        d += 1
    if tmp > 1:
        factors.append((tmp, 1))
    return factors

def _value_from_factors(factors):
    """Compute n from list of (prime, exponent)."""
    n = 1
    for p, e in factors:
        n *= p ** e
    return n

def _tau_perp_from_exponents(exponents, primes):
    """
    Compute tau_perp for n = product(primes[i] ** exponents[i]).
    tau_perp = tau(n) - k, where k = len(exponents)
    tau(n) = product(e_i + 1)
    """
    if not exponents:
        return 0
    tau = 1
    for e in exponents:
        tau *= (e + 1)
    k = len(exponents)
    return tau - k

def _n_from_exponents(exponents, primes):
    """Compute n from exponents and primes."""
    n = 1
    for i, e in enumerate(exponents):
        if e > 0:
            n *= primes[i] ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    max_val = 10**18
    
    # Strategy: Analytically find optimal exponent vectors.
    # For a given k (number of distinct primes), the optimal exponents a_1 >= a_2 >= ... >= a_k
    # maximize product(a_i + 1) subject to product(p_i^a_i) <= 10^18.
    # 
    # Key insight: For fixed k, the optimal exponents are non-increasing.
    # We can enumerate possible exponent patterns for small k.
    # 
    # For k distinct primes, max possible tau is bounded.
    # tau(n) = product(a_i + 1). To maximize tau - k, we want large tau.
    # 
    # Let's enumerate all possible exponent vectors with non-increasing exponents
    # for the first k primes, checking if the product is <= 10^18.
    
    best_n = 2
    best_score = 0
    
    # Maximum number of distinct primes: 2*3*5*...*47 = 614889782588491410 <= 10^18
    # 2*3*5*...*53 > 10^18
    # So max k = 15 (primes up to 47)
    
    # For each k from 1 to 15, find optimal exponents
    # Optimal exponents are non-increasing: a_1 >= a_2 >= ... >= a_k >= 1
    
    # Precompute prime powers for speed
    # primes[i] ** e for relevant e
    
    # For a given k, we need to find a_1 >= a_2 >= ... >= a_k >= 1
    # such that product(primes[i] ** a_i) <= 10^18
    # maximizing product(a_i + 1) - k
    
    # Since exponents are non-increasing and small for larger k,
    # we can do DFS over exponents.
    
    def dfs_exp(idx, max_exp, current_n, current_tau, exponents):
        """DFS to find optimal exponents for first k primes."""
        nonlocal best_n, best_score
        
        if time.time() > deadline - 5:
            return
        
        # If we've assigned exponents for all k primes
        if idx == k:
            if current_n > 1 and current_n <= max_val:
                score = current_tau - k
                if score > best_score:
                    best_score = score
                    n = 1
                    for i, e in enumerate(exponents):
                        n *= primes[i] ** e
                    best_n = n
            return
        
        # Try exponents from max_exp down to 1
        p = primes[idx]
        for e in range(max_exp, 0, -1):
            new_n = current_n * (p ** e)
            if new_n > max_val:
                continue
            new_tau = current_tau * (e + 1)
            exponents.append(e)
            dfs_exp(idx + 1, e, new_n, new_tau, exponents)
            exponents.pop()
    
    # Try different k values
    for k in range(1, 16):
        if time.time() > deadline - 10:
            break
        
        # Max exponent for first prime: 2^a <= 10^18 => a <= 59
        # But with non-increasing constraint, a_1 is bounded by log2(10^18) ~ 60
        # However, for larger k, a_1 must be small.
        
        # Upper bound on a_1: product of smallest k primes with all exponents 1
        min_n = 1
        for i in range(k):
            min_n *= primes[i]
        
        if min_n > max_val:
            break
        
        # Max a_1: 2^a1 * 3^1 * 5^1 * ... <= 10^18
        # a1 <= log2(10^18 / min_n)
        remaining = max_val // min_n
        max_a1 = int(math.log2(remaining)) + 1
        if max_a1 < 1:
            continue
        # Cap at reasonable value
        max_a1 = min(max_a1, 60)
        
        exponents = []
        dfs_exp(0, max_a1, 1, 1, exponents)
    
    # Verify the best found
    if best_n > 1:
        actual_score = _score(best_n)
        if actual_score > best_score:
            best_score = actual_score
        # best_n is already set
    
    # Local search to refine
    if time.time() < deadline - 5 and best_n > 1:
        start_factors = _factorize(best_n)
        
        # Try small perturbations
        primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        current_factors = list(start_factors)
        current_n = best_n
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current_factors)):
                if time.time() > deadline - 5:
                    break
                p, e = current_factors[i]
                # Try increasing exponent
                new_factors = list(current_factors)
                new_factors[i] = (p, e + 1)
                new_n = _value_from_factors(new_factors)
                if new_n <= max_val:
                    s = _score(new_n)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        best_n = current_n
                        best_score = current_score
                # Try decreasing exponent
                if e > 1:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, e - 1)
                    new_n = _value_from_factors(new_factors)
                    s = _score(new_n)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        best_n = current_n
                        best_score = current_score
            # Try adding a new prime
            for p in primes_pool:
                if any(p == pf for pf, _ in current_factors):
                    continue
                new_factors = current_factors + [(p, 1)]
                new_n = _value_from_factors(new_factors)
                if new_n <= max_val:
                    s = _score(new_n)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        best_n = current_n
                        best_score = current_score
            # Try removing a prime
            if len(current_factors) > 2:
                for i in range(len(current_factors)):
                    new_factors = current_factors[:i] + current_factors[i+1:]
                    new_n = _value_from_factors(new_factors)
                    s = _score(new_n)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        best_n = current_n
                        best_score = current_score
    
    return best_n

# EVOLVE-BLOCK-END
