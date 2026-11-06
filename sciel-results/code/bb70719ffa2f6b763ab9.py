"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    target = Fraction(5, 31)
    
    # Known solution: 1/9 + 1/20 + 1/5580 = 5/31
    # Verify: 1/9 + 1/20 = (20+9)/180 = 29/180
    # 5/31 - 29/180 = (900-899)/5580 = 1/5580. Correct!
    known_solution = [9, 20, 5580]
    best_max = max(known_solution)
    best_solution = known_solution
    
    # Try to find solutions with smaller max denominator
    # Search over possible max denominator C, try to find a valid Egyptian fraction
    # with all denominators <= C and distinct
    
    # For a given max C, we need to find distinct d1 < d2 < ... < dk <= C such that sum(1/di) = 5/31
    # This is a subset sum problem in rational arithmetic
    
    # Strategy: iterate C from lower bound upward, check if a solution exists with max <= C
    # Lower bound: since 5/31 > 1/7, we need at least one denominator <= 7... actually 5/31 ≈ 0.1613
    # 1/7 ≈ 0.1429, 1/6 ≈ 0.1667. So smallest denominator must be <= 6? No: 1/6 > 5/31, 
    # so if we use 1/6, remainder is negative. So smallest denom must be >= 7? 
    # Wait: 5/31 = 0.16129... 1/6 = 0.16667 > 5/31. So we can't use 1/6 alone or with others (all positive).
    # Actually we CAN use 1/6 if the sum of others makes it work... no, all terms are positive, so if 1/6 > 5/31, 
    # then 1/6 + positive > 5/31. So denominator 6 is impossible.
    # 1/7 = 0.14286 < 5/31 = 0.16129. Remainder = 5/31 - 1/7 = (35-31)/217 = 4/217 ≈ 0.01843
    # 1/8 = 0.125 < 5/31. Remainder = 5/31 - 1/8 = (40-31)/248 = 9/248 ≈ 0.03629
    # etc.
    
    # Let's do a systematic search for small solutions
    # Try all pairs (a, b) with a < b, both >= 7, and check if 5/31 - 1/a - 1/b is a unit fraction
    # or can be decomposed further with small denominators
    
    # First, check 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # For each a from 7 to some limit, for each b from a+1 to some limit:
    #   rem = 5/31 - 1/a - 1/b
    #   if rem > 0 and rem is a unit fraction 1/c with c > b, we have a solution
    
    # Also check 4-term and more if needed, but 3-term with small c is likely optimal
    
    max_a = 100  # reasonable upper bound for smallest denominator in optimal solution
    max_b = 10000  # upper bound for second denominator
    
    found_better = False
    for a in range(7, max_a + 1):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, max_b + 1):
            if time.time() > deadline - 5:
                break
            # rem = 5/31 - 1/a - 1/b
            # = (5ab - 31b - 31a) / (31ab)
            num = 5 * a * b - 31 * b - 31 * a
            den = 31 * a * b
            if num <= 0:
                # rem <= 0, no good (and for larger b, rem gets more negative or stays <= 0)
                # Actually as b increases, 1/b decreases, so rem increases. 
                # If num <= 0 here, for larger b, num might become positive.
                # Let's not break, just continue
                continue
            
            # Check if rem = num/den is a unit fraction: num/den = 1/c => c = den/num
            if den % num == 0:
                c = den // num
                if c > b:  # distinct and increasing
                    if c <= best_max - 1:  # strictly better
                        best_solution = [a, b, c]
                        best_max = c
                        found_better = True
                    # Even if not better, this is a valid 3-term solution
    
    # Also check if there's a 2-term solution (we showed there isn't, but verify)
    # Already proven no 2-term solution exists
    
    # Check 4-term solutions might give smaller max? Unlikely if 3-term works well,
    # but let's also try greedy-like approaches for small max
    
    # Try to find any solution with max < best_max using a different approach
    # For small C (say up to 5000), do a DFS to find Egyptian fraction with max <= C
    
    if best_max > 1000 and time.time() < deadline - 10:
        # Try DFS for small max denominators
        def dfs(remaining, min_den, max_den, current, depth):
            if time.time() > deadline - 3:
                return None
            if remaining == 0:
                return list(current)
            if remaining < 0:
                return None
            if depth > 6:  # limit depth
                return None
            
            # Lower bound on next denominator: 1/d <= remaining => d >= 1/remaining
            # But also d >= min_den
            # Upper bound: we need remaining - 1/d >= 0 for remaining terms, but also
            # the sum of remaining terms must be achievable. Simple bound: d >= ceil(1/remaining)
            # Actually: 1/d <= remaining => d >= 1/remaining. Also d >= min_den.
            # And d <= max_den.
            
            # Also: if we have (max_depth - depth) terms left, each at most 1/min_possible,
            # the maximum sum we can add is (max_depth - depth) * (1/min_den) roughly.
            # But let's use a simpler bound.
            
            lower = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            upper = max_den
            
            if lower > upper:
                return None
            
            for d in range(lower, upper + 1):
                new_rem = remaining - Fraction(1, d)
                if new_rem < 0:
                    break  # for larger d, 1/d is smaller, so new_rem increases... wait no
                # Actually as d increases, 1/d decreases, so new_rem = remaining - 1/d increases
                # So if new_rem < 0, for smaller d it would be even more negative. 
                # But we're iterating d from lower to upper, so new_rem starts possibly negative and increases.
                # Wait: if d is small, 1/d is large, new_rem = remaining - 1/d could be negative.
                # As d increases, 1/d decreases, new_rem increases.
                # So if new_rem < 0 at some d, for larger d it might become positive.
                # Don't break, just continue.
                
                if new_rem == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                
                if new_rem > 0:
                    current.append(d)
                    result = dfs(new_rem, d + 1, max_den, current, depth + 1)
                    current.pop()
                    if result is not None:
                        return result
            
            return None
        
        # Try various max_den values below best_max
        # We want to find the smallest C such that a solution exists with all denominators <= C
        # Binary search or linear search from a lower bound
        
        # Lower bound: we know a solution exists with max = best_max
        # Try to find smaller
        low = 100
        high = best_max - 1
        
        # Linear search from a reasonable starting point
        # The known solution has max 5580. Let's try to find something smaller.
        # Try C = 200, 300, 400, ..., up to best_max - 1
        
        # Actually, let's just try a few specific values
        for test_max in range(100, min(high + 1, 5000), 50):
            if time.time() > deadline - 5:
                break
            # Try to find a solution with all denominators <= test_max
            # Use DFS starting from denominator 7
            result = dfs(target, 7, test_max, [], 0)
            if result is not None:
                new_max = max(result)
                if new_max < best_max:
                    best_solution = result
                    best_max = new_max
    
    # Return the best solution found
    return best_solution
# EVOLVE-BLOCK-END
