"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from itertools import combinations
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 22
    # Calculate LCM of 1..22 to convert to integer sums
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value
    vals = {x: L // x for x in range(1, n + 1)}
    
    # Sort elements by value descending (smaller x -> larger reciprocal)
    # This tends to create "sparse" sums, which are more likely to be distinct
    elements = sorted(range(1, n + 1), key=lambda x: -vals[x])
    sorted_vals = [vals[x] for x in elements]
    
    best_result = [1]  # fallback
    
    def _is_dissociated_ints(v_list):
        """Check if all subset sums of v_list are distinct using integers."""
        k = len(v_list)
        if k == 0:
            return True
        if k > 20:
            # For very large k, use numpy or assume False? 
            # But we search from top down, so k starts at 22.
            # 2^22 is 4M. Integer set of 4M items might be slow in Python.
            # Let's try to be efficient.
            pass
        
        sums = {0}
        for v in v_list:
            # Generate new sums
            # Optimization: if len(sums) * 2 > some limit, maybe use numpy?
            # But for k <= 20, pure python set is okay.
            # 2^20 = 1M. Set of 1M ints is ~8MB.
            # Loop 20 times, adding 1M items.
            # This might be slow in pure Python (1M * 20 = 20M ops).
            # Let's use a list and sort?
            # No, set is better for collision detection.
            
            # To speed up, we can break early if we detect collision?
            # We can't detect collision until we've generated all sums for the current step?
            # No, collision happens if (sums + v) intersects sums.
            # So we can check intersection before updating?
            # new_sums = sums | {s + v for s in sums}
            # Collision if (sums + v) & sums is not empty.
            
            shifted = {s + v for s in sums}
            if shifted & sums:
                return False
            sums = sums | shifted
            
        return True

    # Search for largest k
    for k in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Use DFS to find a valid subset of size k
        # We use the pre-sorted elements
        # To avoid checking all combinations, we use recursion with pruning
        
        def dfs(start_idx, current_indices, current_vals):
            nonlocal found, best_result
            
            if time.time() > deadline:
                return True # Signal to stop
            
            if len(current_indices) == k:
                # Check if dissociated
                if _is_dissociated_ints(current_vals):
                    best_result = [elements[i] for i in current_indices]
                    found = True
                    return True
                return False
            
            # Pruning: not enough elements left
            if start_idx + (k - len(current_indices)) > n:
                return False
            
            # Try adding elements from start_idx to n - (k - len(current_indices))
            # We want to pick k - len(current_indices) elements from elements[start_idx:]
            
            remaining_needed = k - len(current_indices)
            
            # Heuristic: try smaller indices (larger values) first
            for i in range(start_idx, n - remaining_needed + 1):
                # Add element i
                new_indices = current_indices + [i]
                new_vals = current_vals + [sorted_vals[i]]
                
                # Optional: Prune if current_vals is already not dissociated?
                # But we only check at the end.
                # We can add a light check: if len(current_vals) > 15, check?
                # No, let's stick to end check to keep DFS fast.
                
                if dfs(i + 1, new_indices, new_vals):
                    return True
                
                # If found, stop
                if found:
                    return True
                    
            return False

        # Start DFS
        if dfs(0, [], []):
            break
    
    return best_result
# EVOLVE-BLOCK-END
