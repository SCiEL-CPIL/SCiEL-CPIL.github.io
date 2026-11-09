"""Combined program for packing distinct multiples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    return reduce(_lcm, nums, 1)

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    period = _lcm_all(divisors)  # 30030
    
    deadline = time.time() + 85.0  # Safety margin

    # Pre-sort divisors and their indices to always process the most constrained first?
    # No, the constraint depends on L and W, so the order of candidates changes.
    # But we can pre-compute the LCM.

    # We search for the minimum span W.
    # The minimum possible span is n-1 = 9.
    # We iterate W from 9 upwards.
    # For each W, we iterate L from 1 to period.
    # If a valid assignment is found in [L, L+W], we return it.
    # The first such W is the minimum span.

    for W in range(9, 200):  # Upper bound 200 is safe, likely much smaller
        if time.time() > deadline:
            break
        
        # We only need to check L in [1, period] due to periodicity.
        # However, the window [L, L+W] must be within [1, max_val].
        # Since max_val is 1,000,000 and period is 30,030, 
        # and W is small, L+W will be well within max_val for L in [1, period].
        # L_max = period
        # But we must ensure L >= 1.
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            # Check if window [L, L+W] has a valid assignment
            # Candidates for each divisor d:
            # k_min = ceil(L/d)
            # k_max = floor((L+W)/d)
            
            candidates = []
            feasible = True
            for i, d in enumerate(divisors):
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    feasible = False
                    break
                # Generate candidates
                # Number of candidates is small for small W
                cands = []
                for k in range(k_min, k_max + 1):
                    cands.append(k * d)
                candidates.append(cands)
                if len(cands) == 0:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Backtracking to find a distinct assignment
            # Order indices by number of candidates (MRV heuristic)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            assignment = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                
                i = order[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.discard(v)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # We found a solution for span W.
                # The actual span is max(assignment) - min(assignment).
                # Since we are iterating W from small to large, this is the minimum possible span.
                # Return the list of integers in the order of the original divisors.
                return [int(v) for v in assignment]

    # Fallback if no solution found within time/limits (should not happen)
    # Return the divisors themselves as a valid, though non-optimal, solution
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
