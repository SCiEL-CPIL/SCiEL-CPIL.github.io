"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    n = 256
    # Start with all numbers 2..256
    S = set(range(2, n + 1))
    
    def has_violation(S):
        """Check if there's any triple (a,b,c) distinct in S with a | b*c"""
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return True
        return False
    
    # This is too slow for the full set. Let's use a smarter approach.
    # Instead, let's compute for each element a, how many pairs (b,c) in S have a | b*c
    # and greedily remove the one with the most conflicts.
    
    # But even computing conflicts is O(n^3) per step.
    
    # Let me try a different strategy: start with primes only, then try to add composites.
    
    # First, get all primes
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, isqrt(x) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, n + 1) if is_prime(x)]
    composites = [x for x in range(2, n + 1) if not is_prime(x)]
    
    # Start with all primes
    S = set(primes)
    
    # Try to add composites in some order
    # A composite x can be added if for all distinct b,c in S: x ∤ b*c
    # AND for all distinct a,c in S: a ∤ x*c
    
    # Sort composites by some heuristic - maybe by number of prime factors (fewer is better)
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    # Sort composites by number of prime factors (ascending), then by value
    composites_sorted = sorted(composites, key=lambda x: (num_prime_factors(x), x))
    
    for x in composites_sorted:
        # Check if x can be added to S
        can_add = True
        S_list = list(S)
        
        # Check 1: x ∤ b*c for all distinct b,c in S
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i+1, len(S_list)):
                c = S_list[j]
                if (b * c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            # Check 2: for all distinct a,c in S, a ∤ x*c
            for a in S_list:
                for c in S_list:
                    if a == c:
                        continue
                    if (x * c) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            S.add(x)
    
    return list(S)
# EVOLVE-BLOCK-END
