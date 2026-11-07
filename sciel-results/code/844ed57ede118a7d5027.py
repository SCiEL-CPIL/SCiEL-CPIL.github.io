"""Solve: find 10 distinct positive integers <=1e6, each divisible by its divisor, minimizing range."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Compute LCM of all divisors for periodicity
    lcm_all = reduce(_lcm, divisors)  # 32730
    
    # Deadline guard: 80s safety margin within 90s wall-clock
    deadline = time.time() + 80.0
    
    best_result = None
    best_range = float('inf')
    
    # For a given window [L, L+W], check if we can assign distinct multiples.
    # Order divisors by fewest candidates first (most constrained first).
    
    def can_assign(L, W):
        """Check if we can assign distinct multiples in [L, L+W] for each divisor.
        Returns list of 10 assigned values or None."""
        # For each divisor, compute the list of available multiples in [L, L+W]
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
        
        # Order divisors by number of available multiples (most constrained first)
        order = sorted(range(n), key=lambda i: len(avail[i]))
        
        used = set()
        result = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in avail[i]:
                if v not in used:
                    used.add(v)
                    result[i] = v
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
                    result[i] = 0
            return False
        
        if backtrack(0):
            return result
        return None
    
    # Search over window widths W from 9 upward (theoretical minimum for 10 distinct ints).
    # For each W, check L in [0, lcm_all) since the pattern repeats every lcm_all.
    # The first valid W found is optimal.
    
    for W in range(9, 501):
        if time.time() > deadline:
            break
        
        # For this W, check L values in [0, lcm_all)
        found = False
        for L in range(0, lcm_all):
            if time.time() > deadline:
                break
            
            # Quick feasibility check: each divisor must have at least one multiple in [L, L+W]
            feasible = True
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_max < k_min:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to assign distinct multiples
            result = can_assign(L, W)
            if result is not None:
                # Check all values are positive and <= max_val
                if all(1 <= x <= max_val for x in result):
                    if W < best_range:
                        best_range = W
                        best_result = result[:]
                    found = True
                    break  # Found valid assignment for this W, no need to check more L
        
        # If we found a solution for this W, we can stop (W is increasing)
        if found:
            break
    
    if best_result is None:
        # Fallback: use the divisors themselves (range = 39-6 = 33)
        best_result = divisors[:]
    
    # Ensure types are correct: plain Python ints in a list
    out = [int(x) for x in best_result]
    assert len(out) == n
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == n
    assert all(1 <= x <= max_val for x in out)
    assert all(out[i] % divisors[i] == 0 for i in range(n))
    return out
