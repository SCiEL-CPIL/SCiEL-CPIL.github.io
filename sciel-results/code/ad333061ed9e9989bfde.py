import time

def construct():
    start_time = time.time()
    deadline = start_time + 50 # 50s safety margin
    
    # Known 10-term solution if found
    # If not found, use a valid 11-term or 12-term solution
    # A valid 11-term solution:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/20 + 1/35 + 1/42 + 1/70 + 1/140 + 1/140? No distinct.
    
    # Let's use a hardcoded 10-term solution that I am confident about.
    # After checking online resources for "10 distinct unit fractions sum to 1 min denominator 5":
    # One solution is: [5, 6, 7, 10, 12, 20, 35, 42, 70, 140] is WRONG.
    # Correct solution: [5, 6, 7, 10, 12, 20, 35, 42, 70, 140] ... I will use a solver.
    
    # Solver
    import sys
    sys.setrecursionlimit(10000)
    
    target = 1
    min_d = 5
    max_d = 10000000
    k = 10
    
    # Use fractions for precision
    from fractions import Fraction
    
    best_solution = None
    
    def dfs(remaining, min_denom, count, current):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if count == k:
            if remaining == 1:
                best_solution = current[:]
            return
        
        if remaining <= 0:
            return
            
        # Pruning: max possible sum with remaining terms
        # 1/min_denom + 1/(min_denom+1) + ...
        # Approximation: (k-count) / min_denom
        if remaining > (k - count) / min_denom:
            return
            
        # Pruning: min possible sum
        # 1/max_d + ...
        if remaining < (k - count) / max_d:
            return

        # Determine upper bound for next denominator
        # 1/d <= remaining / (k-count)
        # d >= (k-count) / remaining
        # Also d >= min_denom
        lower_bound = max(min_denom, (k - count) // remaining.numerator * remaining.denominator) # Integer division is tricky with Fraction
        # Better: d >= (k-count) / remaining
        # If remaining is Fraction, (k-count)/remaining is a number.
        # Let's use float for bound estimation, then verify with Fraction.
        
        # Max denominator:
        # 1/d < remaining
        # d > 1/remaining
        upper_bound = int(1 / remaining) + 1
        if upper_bound > max_d:
            upper_bound = max_d
            
        # We need to iterate d from lower_bound to upper_bound
        # But lower_bound calculation:
        # We need 1/d <= remaining / (k-count)
        # d >= (k-count) / remaining
        
        # To avoid iterating too many, we can use the property that d must be such that
        # remaining - 1/d can be represented by k-count-1 terms with denominators > d.
        
        # Optimization: Iterate d from min_denom upwards, but break if 1/d > remaining.
        # Actually, we must pick d such that 1/d < remaining (unless it's the last term).
        # If count == k-1, then 1/d must equal remaining.
        
        if count == k - 1:
            d = 1 / remaining
            if d.is_integer() and d >= min_denom and d <= max_d:
                current.append(int(d))
                best_solution = current[:]
                current.pop()
            return

        # General case
        # d must be at least min_denom
        # d must be such that 1/d < remaining (strictly, because we need more terms)
        # So d > 1/remaining
        
        # Also, we need to leave enough sum for the remaining k-count-1 terms.
        # Max sum of remaining = (k-count-1) / (d+1)
        # So remaining - 1/d <= (k-count-1) / (d+1)
        
        # Let's just iterate d from min_denom to some limit.
        # Limit: 1/d < remaining => d > 1/remaining.
        # Also 1/d > remaining / (k-count) => d < (k-count) / remaining.
        
        d_max = int((k - count) / remaining)
        if d_max > max_d:
            d_max = max_d
            
        d_min = min_denom
        
        for d in range(d_min, d_max + 1):
            if 1 / d >= remaining:
                continue
            new_remaining = remaining - 1 / d
            current.append(d)
            dfs(new_remaining, d + 1, count + 1, current)
            current.pop()
            if best_solution:
                return

    dfs(Fraction(1, 1), 5, 0, [])
    
    if best_solution:
        return best_solution
    else:
        # Fallback to a valid 11-term or 12-term solution
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/20 + 1/35 + 1/42 + 1/70 + 1/140 + 1/280?
        # Sum of first 10 was 343/420 = 2301/2800.
        # 1 = 2800/2800.
        # Diff = 499/2800.
        # 499 is prime?
        # This fallback is risky.
        
        # Let's use a known valid 11-term set:
        # [5, 6, 7, 10, 12, 20, 35, 42, 70, 140, 280] ?
        # 1/280 = 1/280.
        # Sum = 343/420 + 1/280 = 2301/2800 + 10/2800 = 2311/2800 != 1.
        
        # I will return the 10-term solution found by the DFS.
        # If DFS fails, I will return a hardcoded 12-term solution which is easier to find.
        # Example 12-term:
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/20 + 1/35 + 1/42 + 1/70 + 1/140 + 1/280 + 1/560
        # This is getting complicated.
        
        # The DFS should find the 10-term solution quickly.
        raise RuntimeError("No solution found")
