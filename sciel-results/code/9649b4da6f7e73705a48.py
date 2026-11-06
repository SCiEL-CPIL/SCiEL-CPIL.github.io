import time
import sys

# Increase recursion limit just in case, though depth is ~20
sys.setrecursionlimit(10000)

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    containing 1,2,5,11 within [1,284].
    
    Uses Backtracking with Forward Checking (Constraint Propagation).
    """
    prescribed = [1, 2, 5, 11]
    used_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            used_sums.add(x + y)
    
    current = list(prescribed)
    best_set = list(current)
    best_size = len(current)
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Helper to check if adding x is valid
    def is_valid(x, arr, used):
        # Check conflict with existing sums
        # new sums: x + y for y in arr, and 2x
        # Since x > max(arr), all x+y are distinct from each other and from 2x.
        # We just need to check against 'used'.
        for y in arr:
            if x + y in used:
                return False
        if 2 * x in used:
            return False
        return True

    def get_new_sums(x, arr):
        return {x + y for y in arr} | {2 * x}

    def backtrack(arr, used, start_val):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return

        # Pruning: if we can't beat best_size, stop
        # Max possible elements from start_val to 284 is (284 - start_val + 1)
        # But Sidon sets are sparse. A tighter bound is hard to compute on the fly.
        # Simple bound:
        if len(arr) + (284 - start_val + 1) <= best_size:
            return

        if len(arr) > best_size:
            best_size = len(arr)
            best_set = list(arr)

        # Try adding each possible next element
        for x in range(start_val, 285):
            if is_valid(x, arr, used):
                new_used = used | get_new_sums(x, arr)
                backtrack(arr + [x], new_used, x + 1)
                
            # Note: We don't explicitly "skip" x in a way that allows adding x later?
            # No, the loop structure `for x in range(...)` handles skipping by simply not recursing on x,
            # but continuing the loop to x+1.
            # However, this standard DFS explores:
            # 1. Add x, then recurse on x+1...
            # 2. Skip x, try x+1...
            # This is correct.

    # Start backtracking
    backtrack(current, used_sums, 12)
    
    return best_set
