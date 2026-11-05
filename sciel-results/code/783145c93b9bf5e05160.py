"""Improved program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    def update_best(sol):
        nonlocal best_max, best_solution
        if sol and len(sol) <= 256:
            m = max(sol)
            if m < best_max:
                best_max = m
                best_solution = list(sol)
    
    # Meet-in-the-middle for 3-term solutions:
    # Precompute all pairs (d2, d3) with d2 < d3, store sum 1/d2 + 1/d3
    # Then for each d1, check if target - 1/d1 exists in the pair map
    
    # Try increasing max_den bounds
    for max_bound in [100, 200, 300, 400, 500, 800, 1200]:
        if time.time() > deadline:
            break
        if best_max <= max_bound:
            break  # Can't improve
        
        # Build map of pair sums: key = Fraction, value = (d2, d3)
        pair_map = {}
        for d2 in range(2, max_bound + 1):
            if time.time() > deadline:
                break
            for d3 in range(d2 + 1, max_bound + 1):
                s = Fraction(1, d2) + Fraction(1, d3)
                if s >= target:  # Can't be part of valid solution with positive d1
                    continue
                # Store the pair with smallest max(d2,d3) = d3
                if s not in pair_map or d3 < pair_map[s][1]:
                    pair_map[s] = (d2, d3)
        
        # Now try each d1
        # d1 must satisfy: 1/d1 < target, so d1 > 1/target = 31/5 = 6.2, so d1 >= 7
        # Also d1 < max_bound
        for d1 in range(7, max_bound + 1):
            if time.time() > deadline:
                break
            r = target - Fraction(1, d1)
            if r <= 0:
                continue
            if r in pair_map:
                d2, d3 = pair_map[r]
                # Check distinctness: d1, d2, d3 must all be distinct
                if d1 != d2 and d1 != d3 and d2 != d3:
                    # Check ordering (we want distinct denominators, order doesn't matter for sum)
                    sol = sorted([d1, d2, d3])
                    update_best(sol)
                    if best_max <= max_bound:
                        break
        
        if time.time() > deadline:
            break
    
    # If we found a 3-term solution, verify and return
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution):
            return best_solution
    
    # Fallback: greedy Egyptian fraction
    r = target
    values = []
    while r > 0:
        if time.time() > deadline - 1.0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= 0 or d > 10000000:
            break
        if d in values:
            # Try next denominator
            d += 1
            while d in values and d < 10000000:
                d += 1
            if d > 10000000:
                break
        values.append(d)
        r -= Fraction(1, d)
        if len(values) >= 256:
            break
    
    if values and sum(Fraction(1, d) for d in values) == target:
        return values
    
    # Absolute fallback: known solution
    return [8, 31, 248]
# EVOLVE-BLOCK-END
