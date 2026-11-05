import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0 # 85s limit, 5s safety margin
    
    # 1. Compute LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = math.lcm(L, i)
        
    # Weights w_i = L / i
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i

    best_size = 0
    best_set = []
    
    # We will search subsets of indices 1..22
    # To maximize size, we try to include elements.
    # Ordering: 
    # If we process 1..22, we might pick small numbers (large weights).
    # Large weights are more likely to be distinct.
    # However, picking 1, 2, 3, 4, 5... might lead to collisions quickly?
    # Let's just try 1..22 order.
    
    def dfs(idx, current_sums, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we take all remaining, can we beat best?
        remaining = 22 - idx + 1
        if len(current_set) + remaining <= best_size:
            return
            
        if idx > 22:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        w = weights[idx]
        
        # Option 1: Include idx
        # Check collision
        # new_sums = {s + w for s in current_sums}
        # Collision if any s + w is in current_sums
        # Which is equivalent to: (current_sums + w) & current_sums != empty
        
        # Efficient check:
        # Iterate through current_sums and see if (s - w) is in current_sums?
        # Or just generate new sums and check intersection.
        # Given size is small (< 2^15), generating is fine.
        
        has_collision = False
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                has_collision = True
                break
            new_sums.add(ns)
            
        if not has_collision:
            # Include
            dfs(idx + 1, current_sums | new_sums, current_set + [idx])
            
        # Option 2: Exclude idx
        dfs(idx + 1, current_sums, current_set)

    dfs(1, {0}, [])
    
    if best_set:
        return best_set
    else:
        # Fallback to seed if something went wrong
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
