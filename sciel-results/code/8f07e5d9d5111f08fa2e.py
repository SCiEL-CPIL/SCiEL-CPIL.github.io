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
    
    def can_complete(r, min_den, max_den, max_terms=250):
        """Try to complete r using greedy with bounds. Return list or None."""
        if r == 0:
            return []
        values = []
        used = set()
        for _ in range(max_terms):
            if time.time() > deadline:
                return None
            if r == 0:
                return values
            # Need 1/d <= r, so d >= ceil(1/r)
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            if d in used:
                return None
            used.add(d)
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None
    
    # Strategy: try pairs of first two denominators, then greedy complete
    # First denominator d1: 1/d1 < 5/31, so d1 > 31/5 = 6.2, so d1 >= 7
    # Also 1/d1 should not be too small to leave a manageable remainder
    
    for d1 in range(7, 60):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        # Second denominator d2: 1/d2 <= r1, so d2 >= ceil(1/r1)
        d2_min = (r1.denominator + r1.numerator - 1) // r1.numerator
        d2_min = max(d2_min, d1 + 1)
        
        # Try a range of d2 values
        for d2 in range(d2_min, min(d2_min + 50, 500)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            # Try to complete r2 greedily with max_den = best_max - 1
            for max_den in [best_max - 1, 500]:
                if max_den < d2 + 1:
                    continue
                sol = can_complete(r2, d2 + 1, max_den)
                if sol is not None:
                    full_sol = [d1, d2] + sol
                    if len(full_sol) <= 256:
                        m = max(full_sol)
                        if m < best_max:
                            best_max = m
                            best_solution = full_sol
    
    # Also try single first denominator with greedy
    for d1 in range(7, 60):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for max_den in [best_max - 1, 500]:
            if max_den < d1 + 1:
                continue
            sol = can_complete(r1, d1 + 1, max_den)
            if sol is not None:
                full_sol = [d1] + sol
                if len(full_sol) <= 256:
                    m = max(full_sol)
                    if m < best_max:
                        best_max = m
                        best_solution = full_sol
    
    # Bounded DFS for small max_den values
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
            return True
        if depth >= max_depth or r <= 0:
            return False
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try DFS with decreasing bounds
    for search_max in range(min(best_max - 1, 400), 50, -10):
        if time.time() > deadline:
            break
        if dfs(target, 1, search_max, [], 0):
            break
    
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback
    r = target - Fraction(1, 8)
    values = [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
