"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    LIMIT = 10**18
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors
    
    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
            # Factorize n
            factors = []
            temp = n
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if temp % (p * p) > temp:
                    break
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                score += 1
        return score, divisors, factors
    
    def score_factors(factors):
        """Score a factorization without building the full n first."""
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT:
            return -1
        return tau_perp(n, factors)[0]
    
    # Strategy: search over exponent distributions for small primes
    # The key insight is that we want many prime factors with exponent 1,
    # but sometimes higher exponents on smaller primes can help.
    
    # First, let's try the simple case: all exponents are 1 (squarefree)
    # For squarefree n with k distinct primes, n = product of k smallest primes
    # We can fit up to 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    # 16 primes: *53 = 32589158477190044730 > 10^18
    
    # Let's compute the score for squarefree with k primes
    def score_squarefree(k):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        if k > len(primes):
            return -1
        factors = [(primes[i], 1) for i in range(k)]
        return score_factors(factors)
    
    # Try squarefree first
    best_score = 0
    best_factors = None
    for k in range(2, 16):
        s = score_squarefree(k)
        if s > best_score:
            best_score = s
            best_factors = [(p, 1) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53][:k]]
    
    # Now try with some exponents > 1
    # We'll do a search: for each number of distinct primes k, try different exponent patterns
    # Focus on k = 10 to 14 primes with some exponents > 1
    
    primes_small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def search_exponents(k, max_exp=3, deadline=DEADLINE):
        """Search over exponent patterns for k primes."""
        nonlocal best_score, best_factors
        if time.time() > deadline:
            return
        
        # Generate all exponent combinations where at least one is 1
        # and the product is <= LIMIT
        # For efficiency, we'll do a recursive search with pruning
        
        def recurse(idx, current_n, current_exps, current_score_potential):
            nonlocal best_score, best_factors
            if time.time() > deadline:
                return
            
            if idx == k:
                # Compute actual score
                factors = [(primes_small[i], current_exps[i]) for i in range(k)]
                s = score_factors(factors)
                if s > best_score:
                    best_score = s
                    best_factors = factors
                return
            
            # Try exponents 1 to max_exp
            for e in range(1, max_exp + 1):
                new_n = current_n * (primes_small[idx] ** e)
                if new_n > LIMIT:
                    break
                current_exps[idx] = e
                recurse(idx + 1, new_n, current_exps, current_score_potential)
        
        current_exps = [1] * k
        recurse(0, 1, current_exps, 0)
    
    # Search for k = 10 to 14
    for k in range(10, 15):
        if time.time() > DEADLINE - 10:
            break
        search_exponents(k, max_exp=3)
    
    # Also try k = 15 with all exponents 1 (already done) and k = 9 with higher exponents
    for k in range(9, 10):
        if time.time() > DEADLINE - 10:
            break
        search_exponents(k, max_exp=4)
    
    # Try a broader search: allow different prime sets
    # For k primes, we might not want the k smallest primes if we're using higher exponents
    # But for now, let's stick with the smallest primes
    
    # Final check: make sure we have a valid result
    if best_factors is None:
        best_factors = [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1), (17, 1), (19, 1)]
    
    # Build the final number
    n = 1
    for p, e in best_factors:
        n *= p ** e
    
    if n > LIMIT or n < 2:
        n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return n

# EVOLVE-BLOCK-END
