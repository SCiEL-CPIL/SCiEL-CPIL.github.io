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
    
    def check_feasibility(W, L):
        """Check if there is a valid assignment in [L, L+W]."""
        avail = []
        for i, d in enumerate(divisors):
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
            
        # Quick fail: total options must be at least n
        if sum(len(m) for m in avail) < n:
            return None
            
        # Sort indices by number of options (most constrained first)
        order = sorted(range(n), key=lambda i: len(avail[i]))
        
        used = set()
        assignment = [None] * n
        
        def backtrack(idx_in_order):
            if idx_in_order == n:
                return True
            i = order[idx_in_order]
            for m in avail[i]:
                if m not in used:
                    used.add(m)
                    assignment[i] = m
                    if backtrack(idx_in_order + 1):
                        return True
                    used.remove(m)
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Binary search for minimum W
    # We know W=33 works (the divisors themselves: 6..39)
    low, high = 1, 33
    best_result = None
    best_W = 33
    best_L = 1
    
    while low <= high:
        mid = (low + high) // 2
        found = False
        # Iterate L from 1 to lcm_all
        # Optimization: We can break if L + mid > max_val
        for L in range(1, lcm_all + 1):
            if L + mid > max_val:
                break
            res = check_feasibility(mid, L)
            if res is not None:
                found = True
                best_result = res
                best_W = mid
                best_L = L
                break
        if found:
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result is None:
        # Fallback to divisors themselves
        return divisors[:]
        
    # Verify
    assert all(isinstance(x, int) for x in best_result)
    assert len(set(best_result)) == n
    assert all(best_result[i] % divisors[i] == 0 for i in range(n))
    assert all(1 <= x <= max_val for x in best_result)
    
    return list(best_result)
# EVOLVE-BLOCK-END
