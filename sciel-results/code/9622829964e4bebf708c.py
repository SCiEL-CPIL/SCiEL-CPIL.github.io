import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    def check_solution(denoms):
        """Verify a solution is valid and return max denominator."""
        if not denoms:
            return None
        if len(denoms) < 1 or len(denoms) > 256:
            return None
        if len(set(denoms)) != len(denoms):
            return None
        for d in denoms:
            if d < 1 or d > 10000000:
                return None
        total = sum(Fraction(1, d) for d in denoms)
        if total != target:
            return None
        return max(denoms)

    # Verify initial solution
    m = check_solution(best_solution)
    if m is not None:
        best_max_denom = m
    else:
        # Fallback if initial is invalid (shouldn't happen)
        return best_solution

    def find_solution_with_max(k):
        """
        Attempts to find a solution with all denominators <= k.
        Returns the list of denominators if found, else None.
        """
        # Precompute suffix sums of reciprocals for pruning
        # suffix_sum[i] = sum(1/j for j in range(i, k+1))
        suffix_sum = [0.0] * (k + 2)
        for i in range(k, 0, -1):
            suffix_sum[i] = suffix_sum[i+1] + 1.0 / i
            
        def dfs(rem, min_d):
            if time.time() - start_time > time_limit:
                return None
                
            if rem == 0:
                return []
                
            if min_d > k:
                return None
                
            # Pruning 1: The sum of all remaining possible terms must be at least rem
            # If rem > sum(1/d for d in min_d..k), it's impossible
            if float(rem) > suffix_sum[min_d] + 1e-9:
                return None
                
            # Lower bound for d: 1/d < rem => d > 1/rem
            # d >= floor(1/rem) + 1
            # Using float for speed, verified by Fraction subtraction later
            # rem is Fraction. 1/rem is Fraction.
            # min_d must be > 1/rem
            inv_rem = 1 / rem
            # d > inv_rem
            # d >= floor(inv_rem) + 1
            # Note: if inv_rem is integer, d must be > inv_rem, so d >= inv_rem + 1
            # If inv_rem is not integer, d >= ceil(inv_rem)
            # floor(inv_rem) + 1 works for both?
            # If inv_rem = 3.0, floor=3, d>=4. Correct (1/3 < rem is false if rem=1/3, 1/4 < 1/3 true).
            # Wait, if rem = 1/3, 1/d < 1/3 => d > 3 => d >= 4.
            # If rem = 2/5, 1/d < 2/5 => d > 2.5 => d >= 3.
            # floor(2.5) + 1 = 2 + 1 = 3. Correct.
            
            min_d_calc = int(inv_rem) + 1
            if min_d_calc < min_d:
                min_d_calc = min_d
                
            if min_d_calc > k:
                return None
                
            for d in range(min_d_calc, k + 1):
                if time.time() - start_time > time_limit:
                    return None
                    
                next_rem = rem - Fraction(1, d)
                
                if next_rem < 0:
                    # Since d is increasing, 1/d is decreasing.
                    # If 1/d > rem, next_rem < 0.
                    # We should continue to larger d where 1/d becomes smaller.
                    continue
                
                if next_rem == 0:
                    return [d]
                
                # Recurse
                # Next min denom is d+1 (distinct denominators)
                result = dfs(next_rem, d + 1)
                if result is not None:
                    return [d] + result
                    
            return None
            
        return dfs(target, 1)

    # Binary search for the minimum K
    low = 1
    high = best_max_denom
    
    # We know high is feasible.
    # We want to find the smallest K such that find_solution_with_max(K) is not None.
    
    # To speed up, we can check some specific values first if needed, but binary search is O(log N).
    # N = 1736, log2(1736) ~ 11 steps.
    
    while low < high:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        mid = (low + high) // 2
        
        sol = find_solution_with_max(mid)
        
        if sol is not None:
            # A solution exists with max <= mid
            high = mid
            # Update best solution if this one is better (smaller max)
            # The found solution has max <= mid.
            # We should check the actual max of the found solution.
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
        else:
            # No solution found with max <= mid (or timeout)
            # If timeout, we assume infeasible to narrow down, but this is a risk.
            # However, for small mid, search space is small. For large mid, we start from high.
            low = mid + 1
            
    # After binary search, 'high' is the minimum K for which a solution was found in the search.
    # But we should verify the final 'high' one more time to get the solution,
    # in case the last update was skipped or something.
    # Actually, the loop updates best_solution whenever a solution is found.
    # If the loop exits, best_solution should be the best found.
    
    # One final check: try 'high' again?
    # If the binary search ended with low == high, we might not have checked 'high' if it was set by 'low = mid + 1' path?
    # No, if sol is found, high = mid. If not, low = mid + 1.
    # The invariant is: solution exists for 'high'.
    # So 'high' is feasible.
    
    # Let's ensure we have the solution for 'high'
    # It's possible the last successful check was for a larger K, and then we narrowed down.
    # The best_solution variable is updated on every success.
    # So it should hold the solution for the smallest K found.
    
    return best_solution
