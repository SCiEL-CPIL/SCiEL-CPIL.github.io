"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

MAX_N = 256
DEADLINE = 85.0  # Safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    
    # Precompute prime factors for all numbers in 2..256
    pf = {}
    for x in range(2, MAX_N + 1):
        pf[x] = _prime_factors(x)
    
    # A set S is valid iff for every distinct a,b,c in S, a does not divide b*c.
    # Equivalent: for every a in S and every pair of distinct b,c in S\{a},
    #   a | b*c is forbidden.
    # Note: a | b*c iff for every prime power p^k || a, the sum of exponents of p in b and c >= k.
    # But since we only care about divisibility (not exact power), it's simpler to check:
    #   a | b*c  <=>  (b*c) % a == 0
    
    # Strategy: Start with all primes (which are always valid), then greedily add composites
    # that don't create any violation. Then try local search to improve.
    
    # Check if adding candidate x to current set S is valid:
    # For every a in S, and every pair (b,c) in S distinct from a and each other,
    #   a | b*c must be false.
    # Also for the new x: for every a in S, and every pair (b,c) in S (distinct from a, b!=c),
    #   a | b*x must be false, and x | a*b must be false.
    
    def can_add(S, x):
        """Check if x can be added to set S without violating the condition."""
        S_list = list(S)
        # Check: for every a in S_list, for every pair (b,c) from S_list (b!=c, both != a),
        #   a | b*c must be false. (These are already satisfied if S is valid.)
        # New violations involving x:
        # 1. x | a*b for some a,b in S_list with a!=b
        # 2. a | x*b for some a in S_list, b in S_list, a!=b, a!=x (x not in S yet)
        #    i.e., a | x*b where a,b are distinct elements of S_list
        # 3. a | b*x is same as 2.
        # Also: for a in S_list, b,c in S_list with a!=b, a!=c, b!=c: a|b*c (already ok)
        # And: x | a*b (case 1), a | x*b (case 2)
        
        # Case 1: x divides a*b for some distinct a,b in S_list
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if (a * b) % x == 0:
                    return False
        
        # Case 2: a divides x*b for some a in S_list, b in S_list, a != b
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                if (x * b) % a == 0:
                    return False
        
        return True

    # Start with all primes in 2..256
    primes = [x for x in range(2, MAX_N + 1) if _is_prime(x)]
    S = set(primes)
    
    # Try to add composites greedily
    composites = [x for x in range(2, MAX_N + 1) if not _is_prime(x)]
    # Sort composites by some heuristic - maybe by number of prime factors (fewer is "safer")
    composites.sort(key=lambda x: len(pf[x]))
    
    for x in composites:
        if time.time() - start_time > DEADLINE - 10:
            break
        if can_add(S, x):
            S.add(x)
    
    # Local search: try to swap elements to improve
    # Actually, we want to maximize cardinality. Let's try adding more.
    # Try all remaining composites again in different order
    
    # Try a more aggressive approach: try each composite, and if it doesn't fit,
    # try removing one element from S and adding the composite.
    
    best_size = len(S)
    best_set = set(S)
    
    # Try to improve by swapping
    remaining = [x for x in composites if x not in S]
    
    for x in remaining:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try removing each element y from S and adding x
        for y in list(S):
            if y == x:
                continue
            S_new = S - {y}
            if can_add(S_new, x):
                S_new.add(x)
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best_set = set(S_new)
                    S = S_new
    
    # Try adding any remaining composites that now fit
    for x in composites:
        if x not in S:
            if can_add(S, x):
                S.add(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    result = sorted(best_set)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
