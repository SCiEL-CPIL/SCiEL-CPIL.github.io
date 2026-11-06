import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Construct Paley tournament on p=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # in_mask[u] has bit v set if v -> u
    in_mask = [0] * p
    for i in range(p):
        for j in range(p):
            if i != j:
                if (j - i) % p in squares:
                    # i -> j
                    in_mask[j] |= (1 << i)
                else:
                    # j -> i
                    in_mask[i] |= (1 << j)
                    
    # Helper to check if a set of active vertices is valid
    def is_valid(active_list):
        n = len(active_list)
        if n < 4:
            return True
            
        # Create a mask of active vertices
        active_mask = 0
        for v in active_list:
            active_mask |= (1 << v)
            
        # Iterate over all 4-combinations of active_list
        # We use the original indices from active_list
        for v1, v2, v3, v4 in combinations(active_list, 4):
            # Check if there is a common dominator in active_list
            # Dominator v must have v -> v1, v -> v2, v -> v3, v -> v4
            # This means bit v1 is set in in_mask[v], etc.
            # Wait, in_mask[u] has bit v set if v -> u.
            # So we need a vertex v such that:
            # bit v1 is set in in_mask[v]? No.
            # v -> v1 means bit v1 is set in in_mask[v1]? No.
            # in_mask[target] has bit source set if source -> target.
            # So v -> v1 means bit v is set in in_mask[v1].
            # We need a v in active_list such that:
            # bit v is set in in_mask[v1] AND in_mask[v2] AND in_mask[v3] AND in_mask[v4].
            
            # Let M1 = in_mask[v1], M2 = in_mask[v2], etc.
            # We need to find if there is a bit v in active_mask that is set in M1, M2, M3, M4.
            # common_in = M1 & M2 & M3 & M4
            # if (common_in & active_mask) != 0: valid
            # else: invalid
            
            common_in = in_mask[v1] & in_mask[v2] & in_mask[v3] & in_mask[v4]
            if (common_in & active_mask) == 0:
                return False
                
        return True

    current_active = list(range(p))
    
    # Verify initial validity (should be True for Paley 67)
    if not is_valid(current_active):
        # Fallback to a known valid smaller tournament if possible, or return invalid?
        # But Paley 67 is known valid.
        pass

    best_n = len(current_active)
    best_active = list(current_active)
    
    # Greedy removal with multiple restarts
    search_deadline = start_time + 100.0
    
    # We will try to remove vertices in a specific order.
    # To improve, we can sort vertices by some heuristic.
    # Heuristic: Vertices with low "in-degree" or "out-degree" might be less critical?
    # Or just random shuffle.
    
    for attempt in range(20): # Max 20 attempts
        if time.time() > search_deadline:
            break
            
        random.shuffle(current_active)
        
        # Try to remove vertices one by one
        removed_any = True
        while removed_any and time.time() < search_deadline:
            removed_any = False
            for i in range(len(current_active)):
                if time.time() > search_deadline:
                    break
                v = current_active[i]
                # Try removing v
                test_active = current_active[:i] + current_active[i+1:]
                
                if is_valid(test_active):
                    current_active = test_active
                    best_n = len(current_active)
                    best_active = list(current_active)
                    removed_any = True
                    break # Restart the inner loop as indices changed

    # Use the best active set found
    final_active = best_active
    n_out = len(final_active)
    
    # Construct orientation
    # Map old index to new index
    old_to_new = {v: i for i, v in enumerate(final_active)}
    
    orientation = []
    for i in range(n_out):
        for j in range(i+1, n_out):
            old_i = final_active[i]
            old_j = final_active[j]
            # Check direction in original Paley tournament
            # in_mask[old_j] has bit old_i if old_i -> old_j
            if in_mask[old_j] & (1 << old_i):
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': n_out, 'orientation': orientation}
