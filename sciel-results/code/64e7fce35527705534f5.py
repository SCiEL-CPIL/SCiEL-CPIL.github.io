def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    
    # Helper to check if a solution exists with max denominator <= M
    def exists_solution(M, max_terms=8):
        # Use DFS with pruning
        # State: remaining sum, min allowed denominator, max allowed denominator (M)
        # We want to find a set of distinct denominators in [min_d, M] summing to 'rem'
        
        # Precompute prefix sums of reciprocals? 
        # For pruning:
        # Max sum of 'k' terms starting from 'min_d' is sum(1/min_d + 1/(min_d+1) + ...)
        # Min sum of 'k' terms ending at 'M' is sum(1/M + 1/(M-1) + ...)
        
        # To make it fast, we can memoize or just prune aggressively.
        # Since M is bounded in binary search, and max_terms is small, this should be ok.
        
        def dfs(rem, min_d, terms_used):
            if rem == 0:
                return True
            if terms_used >= max_terms:
                return False
            if rem < 0:
                return False
            
            # Pruning 1: Max possible sum with remaining terms
            # We can add at most (max_terms - terms_used) terms.
            # The largest possible sum is using the smallest available denominators: min_d, min_d+1, ...
            max_possible = 0
            d = min_d
            for _ in range(max_terms - terms_used):
                max_possible += Fraction(1, d)
                d += 1
            if rem > max_possible:
                return False
            
            # Pruning 2: Min possible sum with remaining terms
            # The smallest possible sum is using the largest available denominators: M, M-1, ...
            min_possible = 0
            d = M
            for _ in range(max_terms - terms_used):
                if d < 1: break
                min_possible += Fraction(1, d)
                d -= 1
            if rem < min_possible:
                return False
            
            # Iterate possible next denominators
            # We must pick d >= min_d
            # Also 1/d <= rem => d >= 1/rem
            start_d = min_d
            if rem.numerator > 0:
                # d must be at least ceil(1/rem)
                # 1/d <= rem => d >= 1/rem
                # d >= ceil(rem.denominator / rem.numerator)
                lower_bound = (rem.denominator + rem.numerator - 1) // rem.numerator
                if lower_bound > start_d:
                    start_d = lower_bound
            
            # Upper bound for d:
            # If we pick d, remaining is rem - 1/d.
            # We need to be able to form rem - 1/d with remaining terms.
            # Also d <= M.
            
            for d in range(start_d, M + 1):
                # Additional pruning: if 1/d > rem, break (since d is increasing, 1/d decreasing? No, 1/d decreases. So if 1/d < rem, we are good. If 1/d > rem, we skip? No, we need sum = rem. If 1/d > rem, we can't use d because all terms are positive. So we must have 1/d <= rem. This is covered by start_d.)
                
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break # Since d is increasing, 1/d is decreasing, so new_rem will become less negative? 
                    # Wait. rem - 1/d. As d increases, 1/d decreases, so rem - 1/d INCREASES.
                    # So if new_rem < 0, it means 1/d > rem.
                    # For larger d, 1/d is smaller, so rem - 1/d will be larger (closer to rem or positive).
                    # So we should NOT break. We should continue.
                    # Correction:
                    # If d is small, 1/d is large.
                    # If 1/d > rem, then new_rem < 0.
                    # As d increases, 1/d decreases.
                    # Eventually 1/d <= rem.
                    # So we should continue until 1/d <= rem?
                    # Actually, my start_d calculation ensures d >= 1/rem, so 1/d <= rem.
                    # So new_rem should always be >= 0.
                    
                if dfs(new_rem, d + 1, terms_used + 1):
                    return True
            
            return False

        return dfs(target, 7, 0) # Min denom is 7 because 1/6 > 5/31

    # Binary search for the minimum M
    lo = 7
    hi = 100000 # Safe upper bound based on greedy
    best_M = hi
    best_sol = None
    
    # We need to retrieve the solution, not just existence.
    # Modify dfs to return the list of denominators.
    
    def find_solution(M, max_terms=8):
        def dfs(rem, min_d, terms_used, path):
            if rem == 0:
                return list(path)
            if terms_used >= max_terms:
                return None
            if rem < 0:
                return None
            
            # Pruning 1: Max possible sum
            max_possible = 0
            d = min_d
            for _ in range(max_terms - terms_used):
                max_possible += Fraction(1, d)
                d += 1
            if rem > max_possible:
                return None
            
            # Pruning 2: Min possible sum
            min_possible = 0
            d = M
            for _ in range(max_terms - terms_used):
                if d < 1: break
                min_possible += Fraction(1, d)
                d -= 1
            if rem < min_possible:
                return None
            
            start_d = min_d
            if rem.numerator > 0:
                lower_bound = (rem.denominator + rem.numerator - 1) // rem.numerator
                if lower_bound > start_d:
                    start_d = lower_bound
            
            for d in range(start_d, M + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    # This shouldn't happen if start_d is correct, but just in case
                    continue 
                
                path.append(d)
                res = dfs(new_rem, d + 1, terms_used + 1, path)
                if res is not None:
                    return res
                path.pop()
            
            return None

        return dfs(target, 7, 0, [])

    # Binary search
    # We know a solution exists for some M.
    # Let's try to find the smallest M.
    # Note: The search space for DFS can be large if M is large.
    # But pruning is strong.
    
    # To be safe and fast, we can iterate M from a lower estimate?
    # Or binary search.
    # Let's binary search.
    
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        sol = find_solution(mid)
        if sol is not None:
            ans = sol
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if ans is None:
        # Fallback to greedy if binary search fails (should not happen)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return ans
