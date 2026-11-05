"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    target = Fraction(5, 31)
    
    # Search for solution with bounded max denominator
    # Use backtracking with pruning
    
    def solve(max_den, target_frac, min_den, used, depth, max_depth):
        if target_frac == 0:
            return []
        if depth >= max_depth or target_frac < 0:
            return None
        if time.time() > deadline:
            return None
        
        remaining = max_depth - depth
        if remaining <= 0:
            return None
        
        # Prune: max possible sum with remaining terms
        # Each term <= 1/min_den, so max sum = remaining / min_den
        if target_frac > remaining * Fraction(1, min_den):
            return None
        
        # Prune: min possible sum with remaining terms (distinct, >= min_den)
        # The smallest remaining terms would be min_den, min_den+1, ..., min_den+remaining-1
        # Sum of 1/(min_den+i) for i=0..remaining-1
        # This is a lower bound on what we can achieve, but we need exactly target_frac
        # If even the largest possible (using smallest denominators) is less than target, fail
        # Actually we want to know if target is achievable. 
        # The max sum with remaining terms using denominators >= min_den, distinct, <= max_den
        # is achieved by using the smallest available: min_den, min_den+1, etc.
        # But we also have upper bound max_den.
        
        for d in range(min_den, max_den + 1):
            if d in used:
                continue
            term = Fraction(1, d)
            if term > target_frac:
                continue
            new_target = target_frac - term
            if new_target < 0:
                continue
            
            used.add(d)
            result = solve(max_den, new_target, d + 1, used, depth + 1, max_depth)
            used.remove(d)
            
            if result is not None:
                return [d] + result
        
        return None
    
    best_solution = None
    best_max = float('inf')
    
    for max_den in range(10, 100000):
        if time.time() > deadline:
            break
        
        for nt in range(3, 50):
            if time.time() > deadline:
                break
            used = set()
            sol = solve(max_den, target, 1, used, 0, nt)
            if sol is not None:
                s = sum(Fraction(1, d) for d in sol)
                if s == target and len(set(sol)) == len(sol) and all(1 <= d <= 10000000 for d in sol):
                    m = max(sol)
                    if m < best_max:
                        best_max = m
                        best_solution = sol
                    break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback greedy
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
