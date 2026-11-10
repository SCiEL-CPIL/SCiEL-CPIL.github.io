import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = Fraction(5, 31)
    
    # We want to minimize D.
    # Start from a reasonable lower bound.
    # 5/31 approx 0.161.
    # If we use k terms, D is roughly 6.2k.
    # Let's try D from 50 upwards.
    
    best_solution = None
    best_D = 10**8
    
    # We can try to find ANY solution first to have a fallback, 
    # but the task asks to minimize D.
    # The seed solution has max denom around ?
    # Seed: 5/31 - 1/8 = 9/248. 248/9 = 27.5 -> 28. 9/248 - 1/28 = (252-248)/6944 = 4/6944 = 1/1736.
    # Seed solution: [8, 28, 1736]. Max D = 1736.
    # So we can start D from 100 and go up to 1736.
    # If we find a solution with D < 1736, we improve.
    
    # To speed up, we can binary search D if the property "solution exists for D" is monotonic.
    # Is it? If a solution exists for D, does it exist for D+1?
    # Yes, because the set of valid denominators for D is a subset of those for D+1.
    # So we can binary search for the minimum D.
    
    low = 10
    high = 1736 # Seed max denom
    
    def check_D(D):
        # Check if solution exists with max denom <= D
        # Use DFS
        # Max terms: let's say 100 is enough for small D.
        # If D is small, k is small.
        
        # Precompute min/max sums for pruning?
        # Or just calculate on the fly.
        
        def dfs(R, d_min, m, path):
            if time.time() > deadline:
                return None
            
            if R == 0:
                return path
            if m == 0:
                return None
            if R < 0:
                return None
                
            # Lower bound for d
            if R.numerator == 0:
                d_low = d_min
            else:
                d_low = max(d_min, (R.denominator + R.numerator - 1) // R.numerator)
            
            if d_low > D:
                return None
                
            # Upper bound for d
            # We need R' = R - 1/d to be representable.
            # R' >= 0 => 1/d <= R => d >= 1/R.
            
            # Also, R' <= MaxSum(d+1, D, m-1)
            # And R' >= MinSum(d+1, D, m-1)
            
            # Iterate d
            # Optimization: If m-1 is large, the sums converge.
            # For small m, exact sum is fine.
            
            # Heuristic: try smaller d first? Or larger d?
            # Smaller d means larger 1/d, leaving smaller R'.
            # Larger d means smaller 1/d, leaving larger R'.
            # To find ANY solution, order doesn't matter much.
            
            for d in range(d_low, D + 1):
                if R * d < 1:
                    # 1/d > R, so R' < 0.
                    # As d increases, 1/d decreases.
                    # So for larger d, 1/d might be <= R.
                    # Wait, d_low is ceil(1/R). So for d >= d_low, 1/d <= R.
                    # So R' >= 0 is guaranteed.
                    pass
                
                R_prime = R - Fraction(1, d)
                
                # Pruning with bounds
                # Max sum for remaining m-1 terms with min denom d+1
                max_sum = Fraction(0, 1)
                min_sum = Fraction(0, 1)
                
                # Calculate bounds
                # If m-1 is large, sum is approx (m-1)/(d+1) for max, (m-1)/D for min?
                # Let's just sum if m-1 < 20, else use approximations or skip tight bound?
                # For D < 1000, m will be small.
                
                for j in range(m - 1):
                    max_sum += Fraction(1, d + 1 + j)
                    # Min sum uses largest denominators: D, D-1, ...
                    # Denoms must be distinct and > d.
                    # Largest available are D, D-1, ..., D-(m-2).
                    # But we must ensure D-(m-2) > d.
                    # If not, the bound is invalid (impossible to pick m-1 distinct denoms > d and <= D).
                    if D - j <= d:
                        min_sum = None # Impossible to pick enough denoms
                        break
                    else:
                        min_sum += Fraction(1, D - j)
                
                if min_sum is None:
                    continue
                    
                if R_prime > max_sum:
                    continue
                if R_prime < min_sum:
                    continue
                
                res = dfs(R_prime, d + 1, m - 1, path + [d])
                if res is not None:
                    return res
                    
            return None

        # Start DFS
        # What is a good m?
        # For D=100, 5/31 ~ 0.16. 1/100 = 0.01. Need ~16 terms.
        # For D=248, 5/31 ~ 0.16. 1/248 ~ 0.004. Need ~40 terms.
        # Let's try m up to 100.
        # But DFS with m=100 is deep.
        # We can try different m? No, m is just a limit.
        # If we set m=100, it allows up to 100 terms.
        
        # To speed up, we can try to find a solution with fewer terms first?
        # No, we just need ANY solution for this D.
        
        # Let's try m=50 first. If fails, try m=100?
        # Or just m=100.
        
        return dfs(target, 1, 100, [])

    # Binary search for min D
    # check_D is expensive.
    # Maybe linear search from a good guess is faster?
    # Guess: 248 is a candidate.
    # Let's try D from 100 to 300.
    
    # To be safe and fast, let's try a few specific D values or a small range.
    # If we find a solution for D, we can try D-1, D-2...
    
    # Let's try D = 248 first.
    # If it works, we have a score of 248.
    # Then we can try to lower it.
    
    candidate_Ds = list(range(100, 300)) + [1736]
    # Sort by D
    candidate_Ds.sort()
    
    found = None
    for D in candidate_Ds:
        if time.time() > deadline:
            break
        sol = check_D(D)
        if sol is not None:
            found = sol
            best_D = D
            # Can we go lower?
            # We can break and then search lower?
            # Or just continue to find the absolute min in this range?
            # Since we want min D, and we are iterating up, 
            # the first found is the min in this set.
            break
            
    if found is None:
        # Fallback to seed or a known valid solution
        # Seed: [8, 28, 1736]
        return [8, 28, 1736]
        
    return found
