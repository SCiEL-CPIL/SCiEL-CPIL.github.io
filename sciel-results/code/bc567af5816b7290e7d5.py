import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Initial greedy solution to set an upper bound
    # 5/31 = 1/8 + 9/248
    # 9/248 = 1/28 + 4/6944 = 1/28 + 1/1736
    initial_solution = [8, 28, 1736]
    best_max_denom = 1736
    best_solution = initial_solution.copy()
    
    def dfs(remaining, last_denom, current_list):
        nonlocal best_max_denom, best_solution, start_time
        
        if time.time() - start_time > 55:
            return
            
        if remaining == 0:
            max_d = max(current_list)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = current_list.copy()
            return
            
        if len(current_list) >= 256:
            return
            
        # Pruning: if current max is already >= best, we can't improve with this branch
        # unless we haven't added the last term yet? No, max is non-decreasing.
        if current_list and max(current_list) >= best_max_denom:
             # If we are at a leaf or can't improve, prune.
             # Actually, if max(current_list) >= best_max_denom, any extension will have max >= best_max_denom.
             # So we can prune immediately if we are not at the start.
             # But wait, if current_list is empty, max is 0.
             pass 

        a, b = remaining.numerator, remaining.denominator
        # Next denominator d must satisfy 1/d < remaining
        # So d > b/a
        min_d = b // a + 1
        if min_d <= last_denom:
            min_d = last_denom + 1
            
        # Max d we should try
        # If d is too large, the remainder becomes very small, requiring huge subsequent denominators.
        # Also, d must be < best_max_denom to improve the solution.
        max_d = best_max_denom - 1
        
        if min_d > max_d:
            return
            
        # Iterate through possible denominators
        # To find better solutions faster, we might want to explore d values that lead to "nice" remainders.
        # However, simple iteration is robust.
        # We can limit the number of steps to avoid getting stuck in a deep bad branch.
        # But with pruning, it should be fine.
        
        for d in range(min_d, max_d + 1):
            # Check time
            if time.time() - start_time > 55:
                return
                
            # 1/d must be less than remaining
            # This is guaranteed by d >= min_d
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
                
            # Heuristic: if the new remaining fraction has a large numerator/denominator ratio, 
            # it might lead to large denominators.
            # But let's just recurse.
            
            dfs(new_remaining, d, current_list + [d])
            
    dfs(target, 0, [])
    
    return best_solution
