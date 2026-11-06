import math
import time
from math import gcd

_DEADLINE = 110.0  # Conservative deadline to avoid kill
_START = time.time()

def _tau_perp(n):
    """
    Compute the score tau_perp(n): number of adjacent coprime pairs in sorted divisors.
    """
    if n < 2:
        return 0
    
    # Factorize n
    x = n
    factors = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            count = 0
            while x % d == 0:
                x //= d
                count += 1
            factors.append((d, count))
        d += 1 if d == 2 else 2
        
    if x > 1:
        factors.append((x, 1))
        
    # Generate divisors
    divs = [1]
    for p, e in factors:
        new_divs = []
        pk = 1
        for k in range(e + 1):
            for d in divs:
                new_divs.append(d * pk)
            pk *= p
        divs = new_divs
        
    divs.sort()
    
    if len(divs) < 2:
        return 0
        
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    global _START
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    max_n = 10**18
    
    # List of small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    # Strategy: Start with the product of the smallest primes (primorial-like)
    # This maximizes the number of distinct prime factors, which tends to create
    # many coprime adjacent pairs in the divisor lattice.
    
    # Find the largest k such that P_k <= 10^18
    base_n = 1
    base_factors = []
    for p in primes:
        if base_n * p > max_n:
            break
        base_n *= p
        base_factors.append((p, 1))
        
    # If base_n is 1 (shouldn't happen), fallback to 2
    if base_n == 1:
        base_n = 2
        base_factors = [(2, 1)]
        
    best_n = base_n
    best_score = _tau_perp(base_n)
    
    # Local Search: 
    # 1. Try increasing the exponent of the smallest primes if it fits.
    # 2. Try replacing the largest prime factor with a smaller one not already used? No, we used smallest.
    # 3. Try replacing a prime with a slightly larger one to free up space for another prime? 
    #    Actually, we want to MAXIMIZE the count. 
    #    Let's try swapping the largest prime factor p_k with a smaller prime p_j that is not in the set?
    #    No, we started with the smallest. 
    #    
    #    Better heuristic: 
    #    Try to "bump" exponents of the first few primes (2, 3, 5...) to see if it increases the score.
    #    Also try removing the largest prime and adding a smaller one? No, we already have the smallest.
    #    
    #    Let's try a limited random search around the base candidate.
    
    # Generate candidates by modifying exponents of the first 5 primes
    # and by swapping the last prime with a neighbor.
    
    candidate_list = [base_n]
    
    # Modification 1: Increase exponents of small primes
    # We need to be careful not to exceed 10^18.
    # Let's try to add one power of 2, 3, 5, 7, 11 if it fits.
    for i, (p, e) in enumerate(base_factors):
        if p > 11: break # Only tweak small ones
        # Try increasing exponent by 1
        if base_n * p <= max_n:
            new_n = base_n * p
            candidate_list.append(new_n)
        # Try increasing exponent by 2
        if base_n * p * p <= max_n:
            new_n = base_n * p * p
            candidate_list.append(new_n)
            
    # Modification 2: Try replacing the largest prime factor with a smaller prime that was skipped?
    # We didn't skip any small primes. 
    # What if we replace the largest prime P_k with a slightly smaller prime P_j (j < k) that is ALREADY in the set?
    # That would make it not squarefree. That is covered by Modification 1 if we treat it as increasing exponent.
    # But what if we replace P_k with a prime P_j that is NOT in the set but is smaller than P_k?
    # Since we used the *smallest* primes, there are no smaller primes not in the set.
    
    # Modification 3: Try replacing the largest prime P_k with a LARGER prime P_{k+1}?
    # This would allow us to include MORE primes if we drop a smaller one?
    # No, replacing a small prime with a large one reduces the product, allowing us to add MORE primes.
    # Example: Replace 2 with 113? No, 2 is very small.
    # Actually, the product of the first k primes is minimal for k distinct primes.
    # So we can't fit more distinct primes.
    
    # Let's try swapping the largest prime with a slightly larger one, and see if we can fit another prime?
    # No, that makes the product larger.
    
    # What if we drop the largest prime and add two smaller ones?
    # We already have the smallest ones.
    
    # So the main room for improvement is in the EXONENTS.
    # Let's do a quick hill-climbing on exponents for the first few primes.
    
    # We will evaluate a set of promising candidates.
    # To keep it fast, we limit the number of evaluations.
    
    eval_count = 0
    max_evals = 500 # Should be fast enough
    
    # Add some random perturbations
    import random
    random.seed(42)
    
    for _ in range(20):
        # Randomly pick a candidate from the list or base
        curr = base_n
        # Randomly increase/decrease exponents of small primes
        for p in [2, 3, 5, 7, 11]:
            if random.random() < 0.3:
                if curr % p == 0:
                    # Try dividing by p
                    if curr // p >= 2:
                        curr = curr // p
                else:
                    # Try multiplying by p
                    if curr * p <= max_n:
                        curr = curr * p
        if curr > 1:
            candidate_list.append(curr)
            
    # Deduplicate
    candidate_list = list(set(candidate_list))
    
    # Evaluate candidates
    for n in candidate_list:
        if time.time() > deadline - 5:
            break
        if n < 2 or n > max_n:
            continue
        score = _tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
        eval_count += 1
        if eval_count > max_evals:
            break
            
    # If we have time, do a more systematic local search on the best found
    if time.time() < deadline - 10:
        # Try to improve best_n by tweaking exponents of its prime factors
        # Get factors of best_n
        x = best_n
        factors = []
        d = 2
        while d * d <= x:
            if x % d == 0:
                count = 0
                while x % d == 0:
                    x //= d
                    count += 1
                factors.append((d, count))
            d += 1 if d == 2 else 2
        if x > 1:
            factors.append((x, 1))
            
        # Try increasing/decreasing exponents of the first 5 factors
        for i in range(min(5, len(factors))):
            p, e = factors[i]
            # Try e+1
            if best_n * p <= max_n:
                new_n = best_n * p
                score = _tau_perp(new_n)
                if score > best_score:
                    best_score = score
                    best_n = new_n
            # Try e-1 (if e > 1)
            if e > 1 and best_n % p == 0:
                new_n = best_n // p
                score = _tau_perp(new_n)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    
    return best_n
