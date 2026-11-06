import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def dfs(num, den, min_d, current_set):
        nonlocal best_solution, best_max_denom
        
        if time.time() - start_time > time_limit:
            return
        
        if num == 0:
            # Found a solution
            if max(current_set) < best_max_denom:
                best_max_denom = max(current_set)
                best_solution = list(current_set)
            return
        
        if num < 0:
            return
            
        # Lower bound for next denominator d
        # 1/d <= num/den => d >= den/num
        lower_bound = (den + num - 1) // num
        if lower_bound < min_d:
            lower_bound = min_d
            
        # Pruning: if the smallest possible next denominator is already >= best_max_denom,
        # any solution found will have max >= best_max_denom, so we can stop.
        if lower_bound >= best_max_denom:
            return

        # Upper bound heuristic:
        # We need to form num/den with denominators >= lower_bound.
        # The maximum number of terms is limited by 256 - len(current_set).
        # But a tighter bound:
        # If we use d, the remainder is (num*d - den)/(den*d).
        # We need this remainder to be formable by denominators > d.
        # A simple bound: d cannot be too large such that 1/d is much smaller than num/den?
        # Actually, if d is very large, 1/d is very small.
        # Let's just iterate d from lower_bound to best_max_denom - 1.
        # This range is at most 1736, which is small.
        
        for d in range(lower_bound, best_max_denom):
            if time.time() - start_time > time_limit:
                return
            
            # Check if 1/d <= num/den
            if den > num * d:
                continue
                
            next_num = num * d - den
            next_den = den * d
            
            # Reduce fraction
            g = gcd(next_num, next_den)
            if g > 1:
                next_num //= g
                next_den //= g
                
            # Recurse
            dfs(next_num, next_den, d + 1, current_set + [d])
            
    # Start search
    dfs(5, 31, 1, [])

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
