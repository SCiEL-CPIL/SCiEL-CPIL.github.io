def construct():
        import time
        start_time = time.time()
        deadline = start_time + 50 # 10s margin
        
        # Known good solution
        best_solution = [8, 28, 1736]
        best_max = 1736
        
        # Try to find better
        # We suspect a solution exists with max <= 300
        # Let's search for limits 100, 150, 200, 250, 300...
        
        def dfs(rem, min_d, max_d, path):
            # rem: Fraction
            # min_d: next d must be > min_d
            # max_d: absolute max allowed
            # path: list of chosen d
            
            if rem == 0:
                return path[:]
            
            # Pruning
            # Max possible sum from remaining
            # Sum of 1/i for i in (min_d, max_d]
            # Approximation: log(max_d) - log(min_d)
            # But we need exactness? No, just a bound.
            # If rem > sum_{i=min_d+1}^{max_d} 1/i, return None
            
            # Lower bound for next d
            # 1/d < rem => d > 1/rem
            # Also d > min_d
            # And d <= max_d
            
            d_min = max(min_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if d_min > max_d:
                return None
            
            # Upper bound for next d
            # We need enough sum left.
            # If we pick d, remaining rem' = rem - 1/d
            # rem' must be >= 0
            # Also rem' must be representable by denominators in (d, max_d]
            # A simple bound: rem - 1/d <= sum_{i=d+1}^{max_d} 1/i
            
            for d in range(d_min, max_d + 1):
                if time.time() > deadline:
                    return None
                next_rem = rem - Fraction(1, d)
                if next_rem < 0:
                    break # since 1/d decreases as d increases, next_rem will only get larger? 
                    # Wait. rem - 1/d. As d increases, 1/d decreases, so rem - 1/d increases.
                    # If next_rem < 0, it means 1/d > rem.
                    # For larger d, 1/d is smaller, so rem - 1/d will be positive.
                    # So we should NOT break. We should CONTINUE.
                    # However, if 1/d > rem, we can't use this d.
                    # We need 1/d <= rem.
                    # So d >= 1/rem.
                    # My d_min calculation handles this.
                    # So next_rem should be >= 0.
                    
                # Check if next_rem is 0
                if next_rem == 0:
                    return path + [d]
                
                # Check if next_rem is too small to be formed by remaining terms?
                # Min sum with 1 term is 1/max_d.
                # If next_rem < 1/max_d and we need at least 1 more term, fail?
                # No, we can have more terms.
                
                res = dfs(next_rem, d, max_d, path + [d])
                if res:
                    return res
            return None

        from fractions import Fraction
        target = Fraction(5, 31)
        
        # Try limits
        for limit in [100, 150, 200, 250, 300, 400, 500]:
            if time.time() > deadline:
                break
            # Reset time check inside DFS?
            # The DFS above uses global deadline.
            res = dfs(target, 0, limit, [])
            if res:
                max_d = max(res)
                if max_d < best_max:
                    best_solution = res
                    best_max = max_d
                # If we found one, we can stop searching for smaller limits?
                # We are iterating limits increasing. So first found is best?
                # Yes, if we iterate 100, 150, 200... the first valid limit gives the smallest max_d?
                # Not necessarily. A solution with max_d=120 might exist, but DFS with limit=100 fails.
                # DFS with limit=150 finds it. Max_d is 120.
                # So we should record the actual max_d of the solution found.
                # And we can stop if we find a solution with max_d close to limit?
                # Or just keep searching?
                # If we find a solution with max_d=120 using limit=150, we can't do better than 120?
                # We could have a solution with max_d=110?
                # But limit=100 failed. So no solution with max_d <= 100.
                # So 120 is the best so far.
                # We can break if we want to save time, but maybe there's a 110?
                # No, if limit=100 failed, there is no solution with all d <= 100.
                # So max_d must be > 100.
                # So if we found max_d=120, it's the best possible?
                # Yes, because any better solution would have max_d <= 100 (if we had checked) or between 101 and 119.
                # But limit=100 covers all d <= 100.
                # So if limit=100 fails, min possible max_d is >= 101.
                # If limit=150 finds a solution with max_d=120, is it possible there is one with max_d=110?
                # Yes. But limit=100 failed, so 110 is possible.
                # But our DFS with limit=150 might have found 120 first.
                # To be sure, we should iterate limits more finely or just accept the first found?
                # Given time, accepting the first found in increasing limits is a good heuristic.
                break
        
        return best_solution
