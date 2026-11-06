"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    divisors = set()
    for a in range(11):  # 2^a
        for b in range(6):  # 3^b
            for c in range(3):  # 5^c
                for d in range(2):  # 7^d
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val > 12:
                        divisors.add(val)
    
    # Sort divisors
    sorted_divs = sorted(divisors)
    
    best_max = target  # initial best is just [target]
    best_factors = [target]
    
    # DFS: pick distinct factors from sorted_divs in increasing order
    # factors: list of chosen factors so far
    # start_idx: index in sorted_divs to start from (to ensure distinct and ordered)
    # current_product: product of chosen factors so far
    def dfs(factors, start_idx, current_product):
        nonlocal best_max, best_factors
        if time.time() > deadline:
            return
        
        remaining = target // current_product
        if remaining == 1:
            # Found a valid decomposition
            mx = max(factors)
            if mx < best_max:
                best_max = mx
                best_factors = list(factors)
            return
        
        if remaining <= 1:
            return
        
        # Pruning: if we've used too many factors
        if len(factors) >= 12:
            return
        
        # For each possible next factor (must be > 12, distinct, and divide remaining)
        for i in range(start_idx, len(sorted_divs)):
            if time.time() > deadline:
                return
            d = sorted_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            # Additional pruning: if d is too large, the max would exceed current best
            if d >= best_max:
                break
            dfs(factors + [d], i + 1, current_product * d)
    
    dfs([], 0, 1)
    
    return best_factors
# EVOLVE-BLOCK-END
