import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Known good solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    # Search for a solution with max denominator < best_max_denom
    # We allow up to 5 terms to be safe, but 3 or 4 are likely sufficient.
    # Let's search for 3, 4, 5 terms.
    
    def find_solution(max_denom_limit):
        # Returns a list of denominators or None
        # We search for distinct denominators d1 < d2 < ... < dk <= max_denom_limit
        # such that sum(1/di) = target
        
        # Pruning helper
        def dfs(last_denom, current_sum, path, max_terms):
            if time.time() > deadline:
                return None
                
            if current_sum > target:
                return None
                
            if current_sum == target:
                if len(path) >= 1: # At least 1 term
                    return path[:]
                return None

            # If we have used max_terms, we can't add more
            if len(path) >= max_terms:
                return None
                
            # Pruning: Even if we add the smallest possible remaining denominators,
            # can we reach the target?
            # Smallest remaining denom is last_denom + 1.
            # Max possible sum from remaining terms (max_terms - len(path))
            # using smallest available denominators.
            # This is a lower bound on the denominator? No, upper bound on the sum.
            # If max_possible_sum < target - current_sum, prune.
            
            # However, calculating max_possible_sum exactly is hard.
            # Simple bound: 1/(last_denom+1) + 1/(last_denom+2) + ...
            # This is an upper bound on the sum of reciprocals of distinct ints > last_denom.
            # If this upper bound < remaining_target, prune.
            
            remaining_target = target - current_sum
            if remaining_target <= 0:
                return None
                
            # Estimate max add
            # We can add at most (max_terms - len(path)) terms.
            # The largest possible sum is using the smallest available denominators.
            # d_min = last_denom + 1
            # Sum approx k / d_min
            # If k / (last_denom + 1) < remaining_target, it's impossible?
            # No, 1/d is convex. Sum of reciprocals is maximized by smallest denominators.
            # So if sum of smallest k reciprocals < remaining_target, impossible.
            
            k_needed = max_terms - len(path)
            # Approximation: 1/(last+1) + ... 
            # For small k, just sum them.
            max_add = Fraction(0)
            next_d = last_denom + 1
            for _ in range(k_needed):
                if next_d > max_denom_limit:
                    break
                max_add += Fraction(1, next_d)
                next_d += 1
                
            if max_add < remaining_target:
                return None

            # Try next denominator
            # Lower bound for next_d:
            # 1/next_d < remaining_target => next_d > 1/remaining_target
            # Also next_d > last_denom
            # Also next_d <= max_denom_limit
            
            min_next_d = last_denom + 1
            # If 1/min_next_d >= remaining_target, we might need larger d?
            # No, if 1/d >= remaining_target, we can pick it if it doesn't exceed.
            # But if 1/d > remaining_target, we skip.
            
            # Start from max(last_denom + 1, floor(1/remaining_target) + 1) ?
            # If 1/d <= remaining_target, d >= 1/remaining_target.
            # If 1/d > remaining_target, d < 1/remaining_target.
            # We can pick d such that 1/d > remaining_target? No, sum would exceed.
            # So we must have 1/d <= remaining_target.
            # So d >= ceil(1/remaining_target).
            
            lower_bound = int(1 / remaining_target) + 1
            if lower_bound < min_next_d:
                lower_bound = min_next_d
                
            # Also d <= max_denom_limit
            
            for d in range(lower_bound, max_denom_limit + 1):
                next_sum = current_sum + Fraction(1, d)
                if next_sum > target:
                    # Since d increases, 1/d decreases, so next_sum decreases.
                    # We can break? No, we started from lower bound.
                    # If next_sum > target, larger d will give smaller next_sum.
                    # So we continue.
                    pass
                
                res = dfs(d, next_sum, path + [d], max_terms)
                if res is not None:
                    return res
                    
            return None

        # Try 3 terms
        res = dfs(6, Fraction(0), [], 3)
        if res: return res
        
        # Try 4 terms
        res = dfs(6, Fraction(0), [], 4)
        if res: return res
        
        # Try 5 terms
        res = dfs(6, Fraction(0), [], 5)
        if res: return res
        
        return None

    # Search for better solution
    # We know 1736 is possible. Let's try to find one with max denom < 1736.
    # We can binary search or just try limits.
    # Given the small size, we can just try to find ANY solution with max denom < 1736.
    # If we find one, we can try to minimize it further?
    # The task is to minimize the LARGEST denominator.
    # So we want the global minimum.
    
    # Let's try to find the minimal max denominator.
    # We can iterate M from 7 upwards and check if a solution exists with max denom M.
    # But M is the max denom, so all denoms <= M.
    
    # Start from a reasonable guess.
    # We found 1736.
    # Let's check if M=1000 works.
    # If not, increase M.
    
    # However, the DFS above takes max_denom_limit.
    # Let's try to find a solution with max_denom_limit = 1735.
    
    # To be safe and fast, we will try a few specific limits or binary search.
    # But binary search on M requires the property: if solution exists for M, it exists for M+1.
    # This is true.
    
    # Let's binary search M in [7, 1736].
    
    low = 7
    high = 1735 # We know 1736 works, so if we find one < 1736, we can try to lower it.
    # Actually, we want the minimum M.
    # Let's find the smallest M such that a solution exists.
    
    # But the search space is huge.
    # Let's just try to find a solution with M < 1736.
    # If we find one, say M_found, we can then try to find one with M < M_found.
    
    # Given the time limit, let's just run the DFS with limit 1735.
    # If it finds one, we return it.
    # If not, we return 1736 solution.
    
    # To improve, we can try to find the *optimal* M.
    # Let's try M=100, 200, 500, 1000, 1735.
    
    # Actually, the DFS with pruning is fast for small M.
    # Let's try to find the minimum M.
    
    # We'll iterate M from 7 to 1735.
    # But we can skip many.
    # Let's just try to find a solution with M < 1736.
    
    res = find_solution(1735)
    if res:
        return res
        
    # If not found, maybe 1736 is optimal?
    # Or maybe we need more terms?
    # The DFS allows up to 5 terms.
    # If 5 terms doesn't work for M=1735, it's unlikely that 6 terms will work for M<1736?
    # Actually, more terms allow smaller denominators.
    # Let's try 6 terms with M=1735.
    
    # Modify find_solution to take max_terms
    # I hardcoded 3,4,5. Let's make it flexible.
    
    # Re-implement find_solution to try up to 10 terms.
    
    def find_solution_flexible(max_denom_limit, max_terms_limit=10):
        def dfs(last_denom, current_sum, path):
            if time.time() > deadline:
                return None
                
            if current_sum > target:
                return None
                
            if current_sum == target:
                if len(path) >= 1:
                    return path[:]
                return None

            if len(path) >= max_terms_limit:
                return None
                
            remaining_target = target - current_sum
            if remaining_target <= 0:
                return None
                
            # Pruning
            k_needed = max_terms_limit - len(path)
            max_add = Fraction(0)
            next_d = last_denom + 1
            for _ in range(k_needed):
                if next_d > max_denom_limit:
                    break
                max_add += Fraction(1, next_d)
                next_d += 1
                
            if max_add < remaining_target:
                return None

            lower_bound = int(1 / remaining_target) + 1
            min_next_d = last_denom + 1
            if lower_bound < min_next_d:
                lower_bound = min_next_d
                
            for d in range(lower_bound, max_denom_limit + 1):
                next_sum = current_sum + Fraction(1, d)
                if next_sum > target:
                    continue
                
                res = dfs(d, next_sum, path + [d])
                if res is not None:
                    return res
                    
            return None

        return dfs(6, Fraction(0), [])

    # Try to find solution with M < 1736 using up to 10 terms
    res = find_solution_flexible(1735, 10)
    if res:
        return res
        
    return best_solution
