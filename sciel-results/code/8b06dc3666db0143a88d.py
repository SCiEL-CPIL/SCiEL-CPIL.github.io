"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    t0 = time.time()
    TL = 55.0
    
    target = Fraction(5, 31)
    # Known baseline: 1/8 + 1/28 + 1/1736 = 5/31
    # Check: 1/8 + 1/28 = (7+2)/56 = 9/56. 5/31 - 9/56 = (280-279)/1736 = 1/1736. Yes.
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    # Euler-Mascheroni constant
    gamma = 0.5772156649015329
    
    def harm(n):
        if n <= 0:
            return 0.0
        return math.log(n) + gamma + 1.0/(2.0*n) - 1.0/(12.0*n*n)

    def feasible(rem, min_d, max_d):
        """
        Check if 'rem' can be formed by sum of distinct reciprocals from (min_d, max_d].
        Prunes if max possible sum < rem.
        """
        if rem == 0:
            return True
        if rem < 0:
            return False
        
        # 1/d <= rem => d >= 1/rem
        d_min_req = math.ceil(1.0 / float(rem))
        d_start = max(min_d + 1, d_min_req)
        
        if d_start > max_d:
            return False
        
        # Max sum is H(max_d) - H(d_start - 1)
        # If this is less than rem, it's impossible.
        # Note: float(rem) might have precision issues, but rem is Fraction.
        # Using float(rem) is generally safe for comparison with approximations if we use epsilon.
        max_sum = harm(max_d) - harm(d_start - 1)
        if max_sum < float(rem) - 1e-12:
            return False
        return True

    def search(rem, min_d, max_d, path):
        """
        DFS to find a set of denominators.
        Returns list of denominators or None.
        """
        if time.time() - t0 > TL:
            return None
            
        if rem == 0:
            return path
        
        if rem < 0:
            return None
            
        if not feasible(rem, min_d, max_d):
            return None
            
        d_min = max(min_d + 1, math.ceil(1.0 / float(rem)))
        d_max = max_d
        
        if d_min > d_max:
            return None
            
        # Optimization: If we have enough terms left to reach 256 max (not really relevant here, max 256 is huge)
        # We just search.
        
        # Heuristic: Try smaller d first.
        # This tends to find solutions with smaller max denominator faster?
        # Actually, to minimize the MAX denominator, we want to use large denominators early?
        # No, we are bounding the max denominator by max_d in the binary search.
        # Within a fixed max_d, we just need *any* solution.
        
        for d in range(d_min, d_max + 1):
            if time.time() - t0 > TL:
                return None
            
            # Optimization: If 1/d is too small to help? No, we need to sum exactly.
            # If 1/d > rem, skip. But d >= 1/rem ensures 1/d <= rem.
            
            nrem = rem - Fraction(1, d)
            
            if nrem == 0:
                return path + [d]
            
            # Recurse
            res = search(nrem, d, max_d, path + [d])
            if res is not None:
                return res
                
        return None

    # Binary search for the minimum K
    low = 1
    high = best_max_den
    
    while low <= high and time.time() - t0 < TL:
        mid = (low + high) // 2
        
        # We want to find if there is a solution with all denominators <= mid.
        # The search function enforces max_d = mid.
        sol = search(target, 0, mid, [])
        
        if sol is not None:
            # Found a solution.
            # Verify it is valid (sum == 5/31, distinct, in range)
            # The search function ensures distinctness (increasing order) and range.
            # Sum is exact by construction.
            if max(sol) < best_max_den:
                best_max_den = max(sol)
                best_solution = sol
            high = mid - 1
        else:
            # No solution found.
            # Could be timeout or infeasible.
            if time.time() - t0 > TL:
                break
            low = mid + 1
            
    return best_solution
# EVOLVE-BLOCK-END
