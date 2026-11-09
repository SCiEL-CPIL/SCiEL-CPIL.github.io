"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 256
    # Candidate numbers: 2..256
    candidates = list(range(2, N + 1))
    
    # Precompute prime factorization info
    def prime_factors(n):
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
    
    # Precompute factorizations
    factorizations = {n: prime_factors(n) for n in candidates}
    
    def divides_product(a, b, c):
        """Check if a divides b*c."""
        # a | b*c iff for all primes p, v_p(a) <= v_p(b) + v_p(c)
        fa = factorizations[a]
        fb = factorizations[b]
        fc = factorizations[c]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    def is_valid_set(S):
        """Check if set S is valid: no three distinct a,b,c with a|b*c."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True
    
    # Greedy: start with all numbers, remove problematic ones
    # Better: start with a good initial set and try to add more
    
    # Initial set: all primes (known valid since for prime p, p|bc iff p|b or p|c,
    # and no two distinct primes are divisible by the same prime)
    # Actually primes alone: for distinct primes p,q,r, p|q*r? Only if p=q or p=r, which can't happen.
    # So all primes form a valid set.
    
    primes = [n for n in candidates if all(n % d for d in range(2, isqrt(n) + 1))]
    
    # Try to add composites greedily
    S = set(primes)
    
    # Sort composites by some heuristic (maybe by "difficulty" - how many constraints they create)
    composites = [n for n in candidates if n not in S]
    
    # Heuristic: try composites in order of increasing "constraint impact"
    # A composite n creates constraints with pairs (b,c) where n|b*c
    # Simpler: just try in order and check validity
    
    def can_add(S, x):
        """Check if x can be added to set S."""
        S_list = list(S)
        # For each pair (b,c) in S, check if x | b*c
        # And for each b in S, check if b | x*c for some c in S
        # Actually need: for all distinct a,b,c in S∪{x}, a ∤ b*c
        # New triples involving x:
        # 1. x is 'a': x | b*c for some b,c in S
        # 2. x is 'b': a | x*c for some a,c in S
        # 3. x is 'c': a | b*x for some a,b in S
        
        S_list = list(S)
        n = len(S_list)
        
        # Case 1: x divides b*c
        for i in range(n):
            for j in range(i + 1, n):
                if divides_product(x, S_list[i], S_list[j]):
                    return False
        
        # Case 2: a divides x*c (x is the 'b' role)
        for i in range(n):
            for j in range(i + 1, n):
                if divides_product(S_list[i], x, S_list[j]):
                    return False
                if divides_product(S_list[i], S_list[j], x):
                    return False
        
        return True
    
    # Greedy add composites
    added = []
    for x in composites:
        if time.time() > deadline - 5:
            break
        if can_add(S, x):
            S.add(x)
            added.append(x)
    
    result = list(S)
    result.sort()
    
    # Verify
    if not is_valid_set(result):
        # Fallback to primes only
        result = primes
    
    return result

# EVOLVE-BLOCK-END
