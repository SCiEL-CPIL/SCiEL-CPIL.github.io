def construct():
    import time
    import sys
    from fractions import Fraction

    start_time = time.time()
    # Safety margin: return well before 90s kill
    deadline = start_time + 80.0 

    target = Fraction(5, 31)
    
    # We want to find the smallest M such that there exist distinct denominators
    # d_1 < d_2 < ... < d_k <= M such that sum(1/d_i) = 5/31.
    # We can do a DFS for a fixed M.
    # If DFS(M) succeeds, we have a solution with max denom <= M.
    # We iterate M from 2 upwards.
    
    # Optimization:
    # We know 5/31 = 1/8 + 1/28 + 1/1736. So M=1736 is a valid upper bound.
    # We can also check if there are solutions with smaller M.
    # The greedy algorithm gives 1/7 + 1/55 + ... which has large terms.
    # But maybe there is a 2-term or 3-term solution with small max?
    # 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 155ab - 31a - 31b = 0 
    # (5a-31)(5b-31) = 961. 961 = 31^2.
    # Factors of 961: 1, 31, 961.
    # 5a-31=1 => 5a=32 (no int)
    # 5a-31=31 => 5a=62 (no int)
    # 5a-31=961 => 5a=992 (no int)
    # So no 2-term solution.
    
    # 3-term: 1/a + 1/b + 1/c = 5/31.
    # We will let the DFS find it.
    
    def dfs(rem, min_d, max_d, current_denoms, path):
        """
        Recursively find denominators.
        rem: remaining fraction to represent
        min_d: minimum allowed denominator (must be > previous)
        max_d: global upper bound for any denominator in this path
        current_denoms: list of denominators chosen so far
        path: list of denominators for the current path (for reconstruction)
        Returns True if a solution is found, False otherwise.
        """
        if rem == 0:
            return True
        if rem < 0:
            return False
        if len(current_denoms) >= 256:
            return False
        
        # If rem is positive, we need at least one more term.
        # The smallest possible reciprocal is 1/max_d.
        # If 1/max_d > rem, we can't even use the largest allowed denominator to cover rem?
        # No, 1/d <= rem is required. So d >= 1/rem.
        # Also d <= max_d.
        # So we need 1/max_d <= rem. If 1/max_d > rem, then for all d <= max_d, 1/d > rem.
        # So no solution in this branch.
        if Fraction(1, max_d) > rem:
            return False
            
        # Lower bound for next denominator:
        # 1/d <= rem  => d >= 1/rem => d >= ceil(1/rem)
        # Also d >= min_d
        # Also d <= max_d
        
        # Also, we have k terms left to choose (up to 256 - len(current_denoms)).
        # The maximum sum we can achieve with k terms all <= max_d is k/max_d.
        # If rem > k/max_d, we can't reach rem.
        # k_max = 256 - len(current_denoms)
        # if rem > k_max / max_d: return False
        
        # Let's refine the lower bound for d.
        # We need sum of remaining terms = rem.
        # Let m be the number of remaining terms (including the one we pick now).
        # Actually, we pick d, then rem' = rem - 1/d.
        # We need to be able to represent rem' with remaining terms > d and <= max_d.
        
        # Simple bound:
        # d must be at least min_d.
        # d must be at least ceil(1/rem).
        
        # Upper bound for d in this step:
        # We need to be able to represent the remainder with terms > d.
        # The smallest term we can pick next is d+1.
        # The largest possible sum of remaining terms (if we pick as many as possible, up to 256)
        # is bounded by sum_{i=d+1}^{max_d} 1/i ... this is hard to compute fast.
        # Simpler bound:
        # If we pick d, rem' = rem - 1/d.
        # We need rem' >= 0.
        # Also, we need rem' to be representable by terms in [d+1, max_d].
        # The maximum value of 1/x for x in [d+1, max_d] is 1/(d+1).
        # If we have L terms left, max sum is L/(d+1).
        # If rem' > L/(d+1), impossible.
        
        # Let's iterate d from lower_bound to max_d.
        
        lower_bound = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Optimization: If we are close to the deadline, we might want to prune more aggressively.
        # But for now, let's just do the standard DFS with pruning.
        
        for d in range(lower_bound, max_d + 1):
            if time.time() > deadline:
                return False
            
            # Check if 1/d <= rem
            if Fraction(1, d) > rem:
                continue
                
            rem_new = rem - Fraction(1, d)
            
            # Pruning: Can we represent rem_new with denominators in [d+1, max_d]?
            # Max terms remaining: 256 - (len(current_denoms) + 1)
            terms_left = 256 - (len(current_denoms) + 1)
            if terms_left < 0:
                continue
            
            if rem_new == 0:
                # Found a solution
                path.append(d)
                return True
            
            if rem_new < 0:
                continue
            
            # Pruning check:
            # The largest possible sum we can add with 'terms_left' terms, each >= d+1 and <= max_d.
            # To maximize the sum, we should pick the smallest possible denominators: d+1, d+2, ...
            # But we are limited by max_d.
            # The maximum sum is sum_{i=d+1}^{min(d+terms_left, max_d)} 1/i.
            # If rem_new > this sum, it's impossible.
            
            # Approximation for speed:
            # The sum is roughly terms_left / (d+1) if d is small compared to max_d.
            # More accurately, if d+terms_left <= max_d:
            # sum = H_{d+terms_left} - H_d
            # Else:
            # sum = H_{max_d} - H_d
            
            # Let's use a tighter, cheaper bound.
            # Max possible average reciprocal is 1/(d+1).
            # So max sum <= terms_left / (d+1).
            # If rem_new > terms_left / (d+1):
            #    continue
            
            # This bound is valid because 1/x <= 1/(d+1) for all x >= d+1.
            if rem_new > Fraction(terms_left, d + 1):
                continue
                
            # Also, min possible sum with terms_left terms (if we had to use exactly terms_left) 
            # is not a constraint because we can use fewer terms.
            # But we must use AT LEAST 1 term if rem_new > 0.
            
            # Recursive call
            current_denoms.append(d)
            if dfs(rem_new, d + 1, max_d, current_denoms, path):
                return True
            current_denoms.pop()
            
        return False

    best_solution = None
    
    # We know 1736 works. Let's search upwards from a small M.
    # Start M from 2.
    # If we find a solution for M, that M is the minimum possible max denominator.
    
    # However, DFS can be slow if M is large and no solution exists for that M.
    # We can parallelize or use a smarter heuristic?
    # For now, linear scan of M.
    
    # Optimization: 
    # If we find a solution for M, we return it immediately.
    # The score is 1/M. So smaller M is better.
    
    # Let's try M from 2 up to 1736.
    # If we don't find anything by 1736, we return the known solution [8, 28, 1736].
    
    # Actually, we can just run DFS with max_d = 1736 and find the solution.
    # But DFS with max_d=1736 might find a solution with max_d=1736, but maybe there is one with 100?
    # The DFS as written finds *a* solution, not necessarily the one with the smallest max_d *within* the bound.
    # Wait, the loop `for d in range(...)` goes from small to large.
    # The recursion picks the smallest possible d first.
    # This tends to find solutions with small denominators early.
    # However, it doesn't guarantee that the *global* max denominator of the found solution is minimal.
    # Example: It might find [10, 20, 100] (max 100) before [11, 12, 13] (max 13) if the branch for 10 is explored first.
    # But since we want to MINIMIZE the largest denominator, we should:
    # 1. Iterate M from 2 upwards.
    # 2. For each M, check if a solution exists with all denoms <= M.
    # 3. If yes, return it.
    
    # This is the correct strategy for finding the global optimum.
    
    for M in range(2, 1737):
        if time.time() > deadline - 5.0:
            break
            
        # Check if solution exists with max denom <= M
        # We need to be careful: DFS might be slow for large M if no solution exists.
        # But for small M, the search space is small.
        
        # Pruning improvement: 
        # If sum of reciprocals of 1..M is less than target, no solution.
        # Sum_{i=1}^M 1/i ~ ln(M) + gamma.
        # 5/31 ~ 0.16.
        # ln(M) > 0.16 => M > 1.17. So M>=2 is fine.
        
        path = []
        # We need to pass a fresh list for current_denoms? 
        # The dfs function modifies current_denoms. 
        # Let's use a list for the path and just track length.
        
        # The dfs function above uses `current_denoms` to track length for pruning.
        # It also uses `path` to store the solution.
        # Let's restructure slightly to be cleaner.
        
        def dfs_m(rem, min_d, max_d, depth, path):
            if rem == 0:
                return True
            if rem < 0:
                return False
            if depth >= 256:
                return False
            if Fraction(1, max_d) > rem:
                return False
                
            lower_bound = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            terms_left = 256 - depth
            
            for d in range(lower_bound, max_d + 1):
                if time.time() > deadline:
                    return False
                if Fraction(1, d) > rem:
                    continue
                rem_new = rem - Fraction(1, d)
                if rem_new == 0:
                    path.append(d)
                    return True
                if rem_new < 0:
                    continue
                if rem_new > Fraction(terms_left - 1, d + 1):
                    continue
                path.append(d)
                if dfs_m(rem_new, d + 1, max_d, depth + 1, path):
                    return True
                path.pop()
            return False

        if dfs_m(target, 1, M, 0, path):
            # Found a solution with max denom <= M
            # Since we iterate M upwards, this is the minimal M.
            # Verify validity
            if len(path) > 0 and len(path) <= 256:
                if len(set(path)) == len(path):
                    if all(1 <= d <= 10000000 for d in path):
                        s = sum(Fraction(1, d) for d in path)
                        if s == target:
                            return path
        
    # Fallback to known solution
    return [8, 28, 1736]

