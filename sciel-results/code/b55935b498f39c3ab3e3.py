# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    # Calculate LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i
        
    best_set = []
    best_size = 0
    
    # Heuristic: Ascending order (1, 2, ..., 22)
    # Smaller numbers have more dependencies/collisions.
    # Placing them first allows the search to prune invalid branches earlier,
    # potentially exploring more of the search space (larger sets) within the time limit.
    elements = list(range(1, N + 1))
    
    # Backtracking with incremental sums
    # sums: set of all subset sums of current selection
    def backtrack(idx, current_set, sums):
        nonlocal best_set, best_size
        if time.time() - start_time > timeout:
            return
            
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # Pruning: if remaining elements cannot improve best_size
        if len(current_set) + (N - idx) <= best_size:
            return
            
        if idx == N:
            return
            
        elem = elements[idx]
        w = weights[elem]
        
        # Option 1: Include elem
        # Check if adding w creates any duplicate sums
        # new_sums = {s + w for s in sums}
        # If any s + w is in sums, it's a collision.
        # Also, if any s1 + w == s2 + w, then s1 == s2, which is impossible since sums are distinct.
        # So we just need to check if (sums + w) intersects sums.
        
        # Optimization: Check intersection
        # We can iterate through sums and check if s+w in sums.
        # If collision, we cannot include.
        can_include = True
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                can_include = False
                break
            new_sums.add(ns)
            
        if can_include:
            # Include
            new_sums.update(sums) # Wait, sums is the set of ALL subset sums of previous elements.
            # The new set of sums is sums U {s+w for s in sums}.
            # My loop above generated {s+w for s in sums} into new_sums.
            # So the full new set is sums.union(new_sums).
            # But we need to pass the full set to the next level.
            next_sums = sums.union(new_sums)
            
            backtrack(idx + 1, current_set + [elem], next_sums)
            
        # Option 2: Exclude elem
        backtrack(idx + 1, current_set, sums)

    backtrack(0, [], {0})
    
    if best_set:
        return best_set
    return [22] # Fallback

# EVOLVE-BLOCK-END
