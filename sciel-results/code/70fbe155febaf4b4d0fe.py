"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import ceil, log

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    target = Fraction(5, 31)
    
    # Known good solution from current program
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    def dfs(current_sum, last_den, max_den, path, depth=0):
        """
        Depth-first search for Egyptian fraction.
        Returns a list of denominators if found, else None.
        """
        # Check time
        if time.time() > deadline:
            return None
        
        if current_sum == target:
            if path:
                return path[:]
            return None
        
        if current_sum > target:
            return None
            
        rem = target - current_sum
        if rem <= 0:
            return None if current_sum != target else path[:]
            
        # Min next denominator
        min_next = last_den + 1
        if min_next > max_den:
            return None
            
        # Check if max possible sum is sufficient
        # Sum of 1/i for i=min_next..max_den
        # Loose upper bound: (max_den - min_next + 1) / min_next
        # Tighter upper bound: ln(max_den) - ln(min_next-1) approx
        # Let's use the loose one first, it's fast.
        if (max_den - min_next + 1) / min_next < float(rem):
             return None
             
        # Calculate lower bound for d
        # 1/d <= rem => d >= 1/rem
        # rem is Fraction
        # d >= rem.denominator / rem.numerator
        lb = (rem.denominator + rem.numerator - 1) // rem.numerator
        if lb < min_next:
            start_d = min_next
        else:
            start_d = lb
            
        # Optimization: If we are deep, we can limit the range of d
        # to avoid exploring too many branches that are unlikely to work.
        # However, for small max_den (like < 100), the range is small anyway.
        
        # Heuristic: Try denominators that divide the numerator of rem?
        # Or just iterate.
        
        # We iterate from start_d to max_den
        # But we can also use a "greedy" hint: 
        # The greedy algorithm picks d = ceil(1/rem).
        # Let's try that first, then neighbors, then the rest?
        # Actually, simple iteration is fine for small M.
        
        # One more pruning: 
        # If we have k terms left to pick (max depth - depth), 
        # and the max sum of the smallest k terms (starting from min_next) 
        # is less than rem, we fail.
        # But we don't know k exactly, we just have max_den.
        # The loop above handles the "max possible sum" check.
        
        # Iterate
        # To find the MINIMUM max_den, we should probably prefer smaller denominators?
        # No, we are searching for ANY solution with max_den <= M.
        # The order of iteration determines which solution we find first.
        # If we find a solution, we return it.
        # Since we iterate M upwards, the first M that yields a solution is the minimum M.
        
        # Let's try a specific order:
        # 1. d = ceil(1/rem) (Greedy choice)
        # 2. d = ceil(1/rem) + 1, ...
        # This often works well.
        
        greedy_d = lb
        if greedy_d < min_next:
            greedy_d = min_next
            
        # Try greedy_d first
        if greedy_d <= max_den:
            new_sum = current_sum + Fraction(1, greedy_d)
            if new_sum <= target:
                if new_sum == target:
                    return path + [greedy_d]
                res = dfs(new_sum, greedy_d, max_den, path + [greedy_d], depth+1)
                if res:
                    return res
                    
        # Then try others
        # To avoid re-checking greedy_d, start from greedy_d+1
        # But what if greedy_d was < min_next? Then start_d = min_next.
        # If greedy_d >= min_next, we already tried it.
        
        # We should try denominators in increasing order?
        # Or maybe try those that are divisors of rem.numerator first?
        # Let's stick to increasing order for determinism and completeness.
        
        for d in range(start_d, max_den + 1):
            if time.time() > deadline:
                return None
                
            if d == greedy_d:
                continue # Already tried
                
            new_sum = current_sum + Fraction(1, d)
            if new_sum > target:
                # Since d increases, 1/d decreases, so new_sum will decrease.
                # So we should NOT break.
                # Wait, if 1/d is too big, we need a LARGER d.
                # So we continue.
                pass
            elif new_sum == target:
                return path + [d]
            else:
                res = dfs(new_sum, d, max_den, path + [d], depth+1)
                if res:
                    return res
                    
        return None

    # Search for M from lower bound upwards
    # Lower bound analysis:
    # If we use 1 term: 5/31 -> 31/5 = 6.2 -> 7. But 1/7 != 5/31.
    # If we use 2 terms: 1/a + 1/b = 5/31. (5a-31)(5b-31)=961.
    # 961 = 31*31.
    # 5a-31=1 => 5a=32 (no).
    # 5a-31=31 => 5a=62 (no).
    # 5a-31=961 => 5a=992 (no).
    # So no 2-term solution.
    
    # If we use 3 terms:
    # We found [8, 31, 248] with max 248.
    # Can we do better?
    # Let's search M from 28 upwards.
    # Why 28?
    # If max_den < 28, say 27.
    # Max sum with distinct denoms <= 27 is 1/2+...+1/27 (approx 3.0).
    # But we need 5/31 ~ 0.16.
    # The constraint is distinctness and exact sum.
    
    # Let's just start from a reasonable lower bound.
    # For 3 terms 1/a+1/b+1/c = 5/31.
    # 1/a <= 5/31 => a >= 31/5 = 6.2 => a >= 7.
    # 1/b <= 5/31 - 1/7 = (35-31)/217 = 4/217 => b >= 217/4 = 54.25 => b >= 55.
    # 1/c <= 5/31 - 1/7 - 1/55 = (5*55 - 31*7)/ (31*55) - 1/55 ...
    # This suggests if we start with 7, b must be large.
    # If we start with 8:
    # 1/b <= 5/31 - 1/8 = (40-31)/248 = 9/248 => b >= 248/9 = 27.5 => b >= 28.
    # 1/c <= 5/31 - 1/8 - 1/28 = (5*224 - 31*28 - 31*8) / (31*224) ...
    # This path led to 248.
    
    # So M must be at least 28 if we don't use 7.
    # If we use 7, M must be at least 55.
    # So we can check M in [28, 54] (excluding 7 effectively by logic or just letting it fail)
    # and M in [55, 247].
    
    # To be safe and thorough, we search M from 28 to best_max_den - 1.
    
    for M in range(28, best_max_den):
        if time.time() > deadline:
            break
            
        # Optimization: If M < 55, we know 7 cannot be the first term.
        # The DFS will handle this naturally by pruning if 1/7 > rem?
        # No, 1/7 < 5/31. So 7 is a valid first term candidate.
        # But if we pick 7, the next term must be >= 55.
        # If M < 55, the DFS will fail for the branch starting with 7.
        # So it's fine.
        
        sol = dfs(Fraction(0), 0, M, [])
        if sol:
            max_d = max(sol)
            if max_d < best_max_den:
                best_solution = sol
                best_max_den = max_d
                # Since we iterate M upwards, the first solution found for this M
                # has max_den <= M.
                # It might have max_den < M?
                # Yes, e.g. if solution is [10, 20, 30] and M=35, max_d=30.
                # But we are looking for the GLOBAL minimum max denominator.
                # If we found a solution with max_d=30 at M=35, 
                # we should have found it (or a better one) at M=30?
                # Not necessarily, because at M=30, the search space is smaller.
                # But if a solution exists with max_den=30, it uses denominators <= 30.
                # So it would be a valid solution in the search for M=30.
                # So if we found it at M=35, we should have found it at M=30.
                # Unless the search at M=30 failed due to pruning or time?
                # If it failed due to time, we might miss it.
                # But we break on first success.
                # So if we succeed at M, the max_den of that solution is the candidate.
                # We update best_max_den.
                # We can break if best_max_den == M?
                # If best_max_den < M, it means we found a solution with smaller max denom.
                # This should have been found in an earlier iteration?
                # Not strictly, because the search order might differ.
                # But logically, if a solution exists with max denom K, it is a subset of the search space for any M >= K.
                # So the first M where a solution is found should have max_den close to M.
                # Let's just update and continue? 
                # No, if we found a solution, its max_den is the value we care about.
                # We want to minimize that.
                # If we found a solution with max_den=50 at M=50, that's optimal for M=50.
                # Can we get max_den=49? We would have had to find it at M=49.
                # So if we didn't find it at M=49, we can't get it now.
                # So we can break after finding a solution?
                # Yes, because we are iterating M in increasing order.
                # The first M that yields a valid solution provides a candidate with max_den <= M.
                # Any solution with max_den < M would have been a valid solution for a previous M' < M.
                # So if we didn't find it then, we won't find it now (assuming deterministic search).
                # So we can break.
                break
                
    return best_solution
# EVOLVE-BLOCK-END
