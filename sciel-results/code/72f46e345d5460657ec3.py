"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _prime_factorization(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divides_product(a, b, c):
    """Check if a divides b*c."""
    return (b * c) % a == 0

def _is_valid_set(S):
    """Check if set S satisfies the constraint for all distinct a,b,c in S."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(candidate, S):
    """Check if adding candidate to S keeps the set valid."""
    S_list = list(S)
    # Check: for all distinct b,c in S, candidate does not divide b*c
    for i in range(len(S_list)):
        b = S_list[i]
        for j in range(len(S_list)):
            if j == i:
                continue
            c = S_list[j]
            if (b * c) % candidate == 0:
                return False
    # Check: for all distinct a,b in S, candidate does not divide a*c where c is another element
    # Wait, we need: for all distinct a,b,c in S ∪ {candidate}, a ∤ b*c
    # This means:
    # 1. For a = candidate: for all distinct b,c in S, candidate ∤ b*c (checked above)
    # 2. For a in S: for all distinct b,c in S ∪ {candidate} with b≠c, b≠a, c≠a:
    #    - if b,c both in S: already valid since S is valid
    #    - if b = candidate, c in S: a ∤ candidate * c
    #    - if c = candidate, b in S: a ∤ b * candidate (same as above)
    for a in S_list:
        for x in S_list:
            if x == a:
                continue
            if (candidate * x) % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s margin
    
    n_max = P['n']
    
    # Generate all numbers and their properties
    numbers = list(range(2, n_max + 1))
    
    # Separate primes and composites
    primes = []
    composites = []
    for num in numbers:
        if all(num % d != 0 for d in range(2, isqrt(num) + 1)):
            primes.append(num)
        else:
            composites.append(num)
    
    # Start with all primes
    S = set(primes)
    
    # Sort composites by a heuristic: prefer those with larger values, 
    # or those with "rare" prime factors
    # Heuristic: sort by number of divisors descending, then value descending
    # Composites with fewer small prime factors are easier to add
    def composite_heuristic(x):
        pf = _prime_factorization(x)
        # Prefer composites with larger smallest prime factor
        min_pf = min(pf.keys())
        # Prefer composites with fewer total prime factors
        total_exp = sum(pf.values())
        return (min_pf, -total_exp, -x)
    
    composites_sorted = sorted(composites, key=composite_heuristic)
    
    # Greedily add composites
    for cand in composites_sorted:
        if time.time() > deadline:
            break
        if _can_add(cand, S):
            S.add(cand)
    
    result = sorted(S)
    return result
# EVOLVE-BLOCK-END
