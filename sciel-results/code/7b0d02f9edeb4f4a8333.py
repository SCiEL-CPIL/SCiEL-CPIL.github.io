import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_masks():
    """
    Precompute for each x in 1..24 the list of masks M such that
    sum(1/y for y in M) == 1/x and x not in M.
    """
    bad = {x: [] for x in range(1, 25)}
    
    # For each x, we need to find subsets of {1..24} \ {x} summing to 1/x.
    # Given the small range, we can just check all subsets? 2^24 is too big.
    # But we established that for x >= 10, Bad_x is likely empty.
    # And for x < 10, the subsets are small or specific.
    # Let's use a DFS to find subsets for each x.
    
    for x in range(1, 25):
        target = 1.0 / x
        # Elements available
        elems = [y for y in range(1, 25) if y != x]
        
        # To make it faster, we can prune.
        # Sort elems by 1/y descending?
        # Actually, let's just use a simple DFS with pruning based on sum.
        
        def dfs(start_idx, current_sum, current_mask):
            if abs(current_sum - target) < 1e-9:
                if current_mask != 0:
                    bad[x].append(current_mask)
                return
            
            # Pruning: if current_sum > target, return
            if current_sum > target + 1e-9:
                return
            
            # If we run out of elements, return
            if start_idx >= len(elems):
                return
            
            # Pruning: if current_sum + sum(1/y for y in elems[start_idx:]) < target, return
            # This is expensive to compute at each step.
            # Instead, we can sort elems by 1/y descending and use a suffix sum of 1/y.
            
            for i in range(start_idx, len(elems)):
                y = elems[i]
                new_sum = current_sum + 1.0 / y
                if new_sum > target + 1e-9:
                    break # Since sorted descending, further y will be smaller 1/y? 
                    # Wait, if sorted by 1/y descending, then 1/y is decreasing.
                    # So if new_sum > target, further additions will be smaller, 
                    # but we are adding ONE element. 
                    # If we skip i, we might find a solution with i+1?
                    # No, if 1/y is decreasing, and current_sum + 1/y > target, 
                    # then current_sum + 1/(y+1) might be <= target.
                    # So we cannot break. We must continue.
                    
                # Include y
                dfs(i + 1, new_sum, current_mask | (1 << (y - 1)))
                
                # Exclude y? 
                # The loop structure handles exclusion by moving to i+1.
                # But we need to ensure we don't miss combinations.
                # Standard subset sum DFS.
                
        # Sort elems by 1/y descending to help pruning?
        # Actually, just iterate.
        # To speed up, we can limit the size of the subset.
        # Max size for x=1 is 3. For x=2 is 12.
        # Let's just run it. It's fast enough for N=24.
        
        # Optimization: Only consider y > x for x > 12?
        # We proved that for x > 12, Bad_x is empty.
        # So we can skip x > 12.
        if x > 12:
            continue
            
        elems.sort(key=lambda y: 1.0/y, reverse=True)
        dfs(0, 0.0, 0)
        
    return bad

def construct():
    start = time.time()
    deadline = start + 80
    
    bad = _precompute_bad_masks()
    
    def is_valid_mask(S_mask):
        for x in range(1, 25):
            if (S_mask >> (x - 1)) & 1:
                for bm in bad[x]:
                    if (bm & S_mask) == bm:
                        return False
        return True

    best_size = 0
    best_mask = 0
    
    # Greedy: start with all elements, remove those causing conflicts?
    # Or start with empty, add elements?
    # Since conflicts are sparse, start with all and remove is better.
    
    current_mask = (1 << 24) - 1
    
    # Iteratively remove elements to resolve conflicts
    # Find a conflict, remove one element from it.
    # To maximize size, remove the element that is in the most conflicts?
    
    # Let's just try a simple local search or greedy removal.
    # Or simply: Try all subsets of size 22, 21, etc?
    # C(24, 22) = 276. Very small.
    # C(24, 23) = 24.
    # So we can just check all subsets of size 24, 23, 22...
    
    from itertools import combinations
    
    # Check size 24
    if is_valid_mask((1 << 24) - 1):
        return list(range(1, 25))
        
    # Check size 23
    for comb in combinations(range(1, 25), 23):
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        if is_valid_mask(mask):
            return list(comb)
            
    # Check size 22
    for comb in combinations(range(1, 25), 22):
        if time.time() > deadline - 5:
            break
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        if is_valid_mask(mask):
            return list(comb)
            
    # Fallback: Greedy
    current_mask = 0
    for x in range(24, 0, -1):
        trial = current_mask | (1 << (x - 1))
        if is_valid_mask(trial):
            current_mask = trial
            
    return [x for x in range(1, 25) if (current_mask >> (x - 1)) & 1]
