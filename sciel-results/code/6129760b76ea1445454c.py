import time
import math
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # Known solutions to start from
    known_solutions = [
        [8, 28, 1736],      # max = 1736
    ]
    
    best_solution = min(known_solutions, key=lambda s: max(s))
    best_max = max(best_solution)
    
    # Target: try to find solutions with smaller max denominator
    # Use binary search on the maximum denominator limit
    
    def can_solve_with_limit(limit, deadline):
        """Try to find any solution for 5/31 with all denominators in [1, limit]"""
        
        def dfs(num, den, min_d, current_set, remaining_terms_hint):
            if time.time() > deadline:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound for next denominator
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Upper bound check: max possible sum from lower_bound to limit
            # Sum of 1/k for k from lower_bound to limit
            # Approximate: ln(limit) - ln(lower_bound - 1) for large values
            if lower_bound == 1:
                max_possible = math.log(limit) + 0.5772  # H(limit) approx
            else:
                max_possible = math.log(limit) - math.log(lower_bound - 1)
            
            if max_possible < num / den - 1e-12:
                return None
            
            # Try denominators, but be smart about ordering
            # Try values that might lead to clean remainders
            for d in range(lower_bound, min(lower_bound + 500, limit + 1)):
                if time.time() > deadline:
                    return None
                
                if num * d < den:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                if next_num > 0:
                    # Check if remainder is achievable
                    next_lower = d + 1
                    if next_lower > limit:
                        continue
                    if next_lower == 1:
                        max_rem = math.log(limit) + 0.5772
                    else:
                        max_rem = math.log(limit) - math.log(next_lower - 1)
                    if max_rem < next_num / next_den - 1e-12:
                        continue
                
                res = dfs(next_num, next_den, d + 1, current_set + [d], remaining_terms_hint)
                if res is not None:
                    return res
            
            # If we didn't find in the first 500, try some strategic values
            for d in range(lower_bound + 500, limit + 1, max(1, (limit - lower_bound) // 100)):
                if time.time() > deadline:
                    return None
                
                if num * d < den:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                if next_num > 0:
                    next_lower = d + 1
                    if next_lower > limit:
                        continue
                    if next_lower == 1:
                        max_rem = math.log(limit) + 0.5772
                    else:
                        max_rem = math.log(limit) - math.log(next_lower - 1)
                    if max_rem < next_num / next_den - 1e-12:
                        continue
                
                res = dfs(next_num, next_den, d + 1, current_set + [d], remaining_terms_hint)
                if res is not None:
                    return res
            
            return None
        
        return dfs(5, 31, 1, [], 0)
    
    # Try to find solutions with decreasing limits
    # Start from just below current best and work down
    test_limits = []
    # Try key thresholds
    for d in range(best_max - 1, 900, -10):
        test_limits.append(d)
    # Also try specific values
    test_limits.extend([930, 920, 910, 900, 850, 800, 750, 700, 650, 600, 550, 500])
    test_limits = sorted(set(test_limits), reverse=True)
    
    for limit in test_limits:
        if time.time() > time_limit - 5.0:
            break
        
        deadline = min(time.time() + 3.0, time_limit - 5.0)
        sol = can_solve_with_limit(limit, deadline)
        
        if sol is not None:
            actual_max = max(sol)
            if actual_max < best_max:
                best_max = actual_max
                best_solution = sol
                # Continue searching for even better
                continue
            else:
                break
        else:
            # No solution found for this limit, try smaller
            continue
    
    # Verify the solution
    if best_solution:
        total = sum(Fraction(1, d) for d in best_solution)
        assert total == Fraction(5, 31), f"Solution invalid: {total} != 5/31"
        assert len(set(best_solution)) == len(best_solution), "Denominators not distinct"
        assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    
    return best_solution
