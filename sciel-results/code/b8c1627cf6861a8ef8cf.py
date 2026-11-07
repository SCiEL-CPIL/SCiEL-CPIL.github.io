"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Compute LCM of all divisors
    lcm_all = reduce(_lcm, divisors)
    
    def try_assign(L, W):
        """Try to assign distinct multiples in [L, L+W] for each divisor."""
        # For each divisor, get available multiples
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + W) // d       # floor((L+W)/d)
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
        
        # Backtracking to find distinct assignment
        used = set()
        assignment = [None] * n
        
        def backtrack(i):
            if i == n:
                return True
            for m in avail[i]:
                if m not in used:
                    used.add(m)
                    assignment[i] = m
                    if backtrack(i + 1):
                        return True
                    used.remove(m)
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Search for minimum W
    for W in range(1, 500):  # Try small W values
        found = False
        for r in range(lcm_all):
            L = r if r >= 1 else lcm_all
            if L + W > max_val:
                continue
            result = try_assign(L, W)
            if result is not None:
                # Verify
                assert all(isinstance(x, int) for x in result)
                assert len(set(result)) == n
                assert all(result[i] % divisors[i] == 0 for i in range(n))
                assert all(1 <= x <= max_val for x in result)
                return list(result)
    
    # Fallback: return the divisors themselves
    return divisors[:]
# EVOLVE-BLOCK-END
