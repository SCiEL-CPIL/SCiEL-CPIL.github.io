import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    best_diff = float('inf')
    best_L = -1
    best_R = -1
    best_solution = None
    
    # Helper: Bipartite matching to check if distinct representatives exist in [L, R]
    # Returns True if a valid assignment exists, and optionally the assignment
    def check_and_solve(L, R, need_solution=False):
        # Generate candidates for each divisor
        # candidates[i] is a list of multiples of divisors[i] in [L, R]
        candidates = [[] for _ in range(n)]
        
        for i, d in enumerate(divisors):
            # First multiple >= L
            k = (L + d - 1) // d
            x = k * d
            while x <= R:
                candidates[i].append(x)
                x += d
        
        # If any divisor has no candidates, return False
        for i in range(n):
            if not candidates[i]:
                return (False, None) if need_solution else False
        
        # Bipartite matching: Left side = Divisors (0..n-1), Right side = Values
        # We want to assign a unique value to each divisor.
        # Use DFS-based augmenting path (Kuhn's algorithm) which is fine for n=10
        
        match_to_div = {}  # value -> divisor_index
        match_to_val = {}  # divisor_index -> value
        
        def dfs(div_idx, visited):
            for val in candidates[div_idx]:
                if val in visited:
                    continue
                visited.add(val)
                # If val is unmatched, or if the divisor currently matched to val can find another match
                if val not in match_to_div or dfs(match_to_div[val], visited):
                    match_to_div[val] = div_idx
                    match_to_val[div_idx] = val
                    return True
            return False
        
        for i in range(n):
            visited = set()
            if not dfs(i, visited):
                return (False, None) if need_solution else False
        
        if need_solution:
            # Construct the solution list in the order of divisors
            solution = [0] * n
            for i in range(n):
                solution[i] = match_to_val[i]
            return (True, solution)
        else:
            return True

    # Search over L in one period
    # We only need to check L up to period because the pattern repeats.
    # However, the optimal window might start at any L.
    # Actually, since the set of multiples is periodic with period LCM,
    # the relative configuration of multiples within any window [L, L+W]
    # depends only on L mod LCM.
    # So checking L in [1, LCM] is sufficient to find the global minimum span.
    
    # Optimization: We can limit the binary search range for W.
    # The maximum span needed is likely small, but let's be safe.
    # A span of LCM is always sufficient? Not necessarily, but it's a good upper bound.
    # Actually, the maximum gap between multiples of different divisors is bounded.
    # Let's just binary search W in [0, period].
    
    for L in range(1, period + 1):
        if time.time() > deadline:
            break
            
        # Binary search for the minimum W such that [L, L+W] is valid
        # W is the span. R = L + W.
        # We want to minimize W.
        
        # If we already have a best_diff, we don't need to search W >= best_diff
        # because we only care about improving the best.
        lo = 0
        hi = min(period, best_diff - 1) if best_diff < float('inf') else period
        
        # If hi < lo, no improvement possible for this L
        if hi < lo:
            continue
            
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            R = L + mid
            
            # Check if a valid assignment exists in [L, R]
            if check_and_solve(L, R):
                # Found a valid window with span mid
                # Update best
                if mid < best_diff:
                    best_diff = mid
                    best_L = L
                    best_R = R
                    # We can try to get the solution now, or later.
                    # Getting it now is safer to ensure we have a valid solution if time runs out.
                    # But check_and_solve with need_solution=True is slightly slower.
                    # Let's just store L, R and solve at the end.
                    # However, if we break early, we need the solution.
                    # Let's store the best L, R.
                found = True
                hi = mid - 1  # Try smaller span
            else:
                lo = mid + 1  # Need larger span
        
    if best_L != -1:
        # Solve for the best L, R to get the actual values
        valid, solution = check_and_solve(best_L, best_R, need_solution=True)
        if valid:
            # Ensure types are Python ints
            return [int(x) for x in solution]
    
    # Fallback: If the search failed or found nothing (shouldn't happen), 
    # use a simple heuristic: pick the smallest multiple for each divisor?
    # This might not be distinct or minimal, but it's a valid fallback structure.
    # Actually, the problem guarantees a solution exists.
    # If we reach here, it means the search loop didn't find a solution, 
    # which is impossible if the logic is correct.
    # Let's do a brute force fallback just in case.
    
    # Brute force fallback:
    # Pick k_i such that k_i * d_i are distinct and close.
    # Simple approach: iterate span W from 1 upwards, check [1, 1+W], [2, 2+W]...
    # This is slow, but better than crashing.
    
    # Given the constraints, the LCM search should always find a solution.
    # If it didn't, return the divisors themselves (which are distinct multiples of themselves)
    # Wait, divisors are [6, 10, ...]. They are distinct.
    # Max - Min = 39 - 6 = 33.
    # This is a valid solution, though likely not optimal.
    
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
