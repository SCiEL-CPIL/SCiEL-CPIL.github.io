import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM to define the period of the problem
    lcm_all = reduce(_lcm, divisors)
    
    # Deadline guard
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    best_result = None
    best_range = float('inf')
    
    # We know minimum range is 9 (10 distinct ints)
    # We search for increasing range W
    # Since the pattern repeats every lcm_all, we only need to check L in [1, lcm_all]
    # for each W.
    
    # To be safe and fast, we can just check W from 9 upwards.
    # For each W, we check if there's an L such that we can assign distinct multiples.
    
    # Optimization: Precompute which multiples are available for each divisor in a window of size W?
    # No, L changes.
    
    # Let's use a backtracking solver for a fixed L and W.
    
    def can_assign(L, W):
        # candidates[i] is a list of values x_i such that x_i % divisors[i] == 0 and L <= x_i <= L+W
        candidates = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            # first multiple >= L
            k_min = (L + d - 1) // d
            # last multiple <= L+W
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            # generate candidates
            # We want to try smaller candidates first? Or larger?
            # Doesn't matter much for existence, but for distinctness, 
            # sorting by value might help pruning if we track used values.
            vals = [k * d for k in range(k_min, k_max + 1)]
            candidates[i] = vals
        
        used_values = set()
        
        # Order divisors by number of candidates (most constrained first)
        indices = sorted(range(n), key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if idx == n:
                return True
            i = indices[idx]
            for v in candidates[i]:
                if v not in used_values:
                    used_values.add(v)
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(v)
            return False
            
        if backtrack(0):
            # Reconstruct solution
            # We need to know which value was assigned to which index.
            # The above backtrack doesn't store the assignment.
            # Let's modify it to return the assignment.
            pass
        return None

    # Let's rewrite can_assign to return the assignment
    def solve(L, W):
        candidates = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            vals = [k * d for k in range(k_min, k_max + 1)]
            candidates[i] = vals
        
        used_values = set()
        indices = sorted(range(n), key=lambda i: len(candidates[i]))
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = indices[idx]
            for v in candidates[i]:
                if v not in used_values:
                    used_values.add(v)
                    assignment[i] = v
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(v)
            return False
            
        if backtrack(0):
            return assignment
        return None

    # Search
    # We iterate W from 9 upwards.
    # For each W, we iterate L from 1 to lcm_all.
    # If found, return.
    
    # Note: The problem asks to minimize max - min.
    # So we must find the minimum W.
    
    # Is it possible that a larger W allows a smaller range? No, range IS W.
    # So we just find the minimum W for which a solution exists.
    
    # However, we must ensure the values are <= 1000000.
    # If L + W <= 1000000, it's valid.
    # Since lcm_all is ~30000, and we can shift L by multiples of lcm_all,
    # we can always find a small L if a pattern exists.
    # Specifically, if a solution exists for L in [1, lcm_all], it exists for L + k*lcm_all.
    # We just need one L such that max_val is respected.
    # Since min L is 1, and max value is L+W, and L < 30000, W < 100 (likely),
    # max value < 30100, which is well below 1,000,000.
    
    for W in range(9, 1000):  # Cap at 1000 to be safe, but 9-20 is likely enough
        if time.time() > deadline:
            if best_result:
                return best_result
            # If we haven't found anything, return a valid but maybe suboptimal solution
            # e.g. the divisors themselves
            return divisors[:]
            
        # Check L in [1, lcm_all]
        # Optimization: We can break early if we find one.
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            res = solve(L, W)
            if res:
                # Verify constraints
                # Distinct
                if len(set(res)) != n:
                    continue
                # Divisibility
                valid = True
                for i, val in enumerate(res):
                    if val % divisors[i] != 0:
                        valid = False
                        break
                    if val < 1 or val > max_val:
                        valid = False
                        break
                if valid:
                    # This is the minimum W found so far (since we iterate W upwards)
                    # So this is the optimal solution for this W.
                    # Since we want global min W, this is the answer.
                    return res
        
        # If we finished all L for this W, try next W

    # Fallback: return divisors
    return divisors[:]
