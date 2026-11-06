"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    def find_solution(k, max_denom=10000000, min_denom=5):
        """Find k distinct denominators in [min_denom, max_denom] whose reciprocals sum to 1."""
        best = [None]
        
        def dfs(remaining, min_d, chosen, depth):
            if time.time() > deadline:
                return
            if depth == k:
                if remaining == 0:
                    best[0] = chosen[:]
                    return
                return
            if remaining < 0:
                return
            
            # Pruning using exact Fraction arithmetic
            # Max possible sum with (k-depth) terms starting from min_d
            max_possible = Fraction(0)
            for i in range(k - depth):
                max_possible += Fraction(1, min_d + i)
            if max_possible < remaining:
                return
            
            # Min sum: (k-depth) / max_denom
            min_possible = Fraction(k - depth, max_denom)
            if min_possible > remaining:
                return
            
            if remaining <= 0:
                return
                
            # The next denominator d must satisfy:
            # 1/d <= remaining  =>  d >= 1/remaining
            # 1/d >= remaining / (k - depth)  =>  d <= (k-depth) / remaining
            # And d >= min_d
            
            # d >= ceil(1/remaining)
            d_min = max(min_d, -(-remaining.denominator // remaining.numerator) + 1)
            # d <= floor((k-depth) / remaining)
            d_max = min(max_denom, (k - depth) * remaining.denominator // remaining.numerator)
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    continue
                if new_remaining == 0:
                    if k - depth == 1:
                        best[0] = chosen + [d]
                        return
                    continue
                chosen.append(d)
                dfs(new_remaining, d + 1, chosen, depth + 1)
                chosen.pop()
                if best[0] is not None:
                    return
        
        dfs(Fraction(1, 1), min_denom, [], 0)
        return best[0]
    
    # Try k=2, 3, 4, ... until we find a solution
    for k in range(2, 15):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            # Verify with exact arithmetic
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use a known representation or the DP approach
    # Use the DP approach as fallback with exact fractions
    # Try to find a solution using a common denominator approach
    # 1 = sum of 1/d_i, so we need d_i such that sum of LCM/d_i = LCM
    
    # Try a few known decompositions
    known_solutions = [
        [5, 6, 7, 42, 140, 210, 420, 840, 1260, 2520],
        [5, 7, 10, 35, 70, 140, 210, 420, 840, 1680],
        [6, 7, 10, 15, 21, 35, 42, 70, 105, 210],
    ]
    
    for sol in known_solutions:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol
    
    # Use the DP approach as final fallback
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
