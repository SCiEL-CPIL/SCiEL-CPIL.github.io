"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    D = reduce(lcm, range(1, 23), 1)
    values = [D // i for i in range(1, 23)] # values[i-1] is the integer weight for 1/i
    
    best_set = []
    best_size = 0
    
    # Greedy initialization to get a good lower bound
    # Try adding elements from largest to smallest (smallest reciprocals first)
    # Or smallest to largest?
    # Let's try both or just one. Smallest reciprocals (large i) are "finer", 
    # might allow packing more.
    greedy_set = []
    greedy_sums = {0}
    for i in range(22, 0, -1):
        v = values[i-1]
        new_sums = greedy_sums | {s + v for s in greedy_sums}
        if len(new_sums) == 2 * len(greedy_sums):
            greedy_set.append(i)
            greedy_sums = new_sums
    
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
    
    # Backtracking search
    # idx is 1-based index in 1..22
    # current_set is list of elements
    # current_sums is set of integer sums
    def dfs(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning 1: Can't beat best_size
        if len(current_set) + (22 - idx + 1) <= best_size:
            return
            
        # Pruning 2: Collision detected
        # If current_sums has fewer than 2^k elements, it's not dissociated
        # But we maintain the invariant that current_sums is valid for current_set
        # So we only check when we form new_sums
        
        if idx > 22:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        v = values[idx-1]
        
        # Option 1: Include idx
        # Check if adding v creates a collision
        # new_sums = current_sums U (current_sums + v)
        # If len(new_sums) < 2 * len(current_sums), collision
        # We can check this without constructing the full set if we are lucky,
        # but constructing it is O(2^k).
        
        # Optimization: Only construct if we might improve best_size?
        # No, we need to know if it's valid to recurse.
        
        # Heuristic: Check collisions by iterating?
        # For small k, set union is fast.
        # For large k, it's slow.
        
        # Let's just do it.
        new_sums = current_sums | {s + v for s in current_sums}
        
        if len(new_sums) == 2 * len(current_sums):
            current_set.append(idx)
            dfs(idx + 1, current_set, new_sums)
            current_set.pop()
            
        # Option 2: Exclude idx
        dfs(idx + 1, current_set, current_sums)

    dfs(1, [], {0})
    
    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
