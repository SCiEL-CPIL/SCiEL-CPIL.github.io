"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
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
    
    def _is_dissociated_fast(v_list):
        """Check if all subset sums of v_list are distinct using integers.
        Returns True if dissociated, False otherwise."""
        if not v_list:
            return True
        sums = {0}
        for v in v_list:
            shifted = {s + v for s in sums}
            if shifted & sums:
                return False
            sums |= shifted
        return True
    
    def _greedy_build():
        """Greedy: add elements one by one if they keep the set dissociated."""
        result = []
        result_sums = {0}
        for i in range(n):
            v = sorted_vals[i]
            shifted = {s + v for s in result_sums}
            if not (shifted & result_sums):
                result.append(i)
                result_sums |= shifted
        return result
    
    # First try greedy to get a good starting point
    greedy_result = _greedy_build()
    if len(greedy_result) > len(best_result):
        best_result = [elements[i] for i in greedy_result]
    
    # Now search for larger sets using DFS with incremental pruning
    best_k = len(best_result)
    
    for k in range(n, best_k, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        def dfs(start_idx, current_indices, current_sums):
            nonlocal found, best_result, best_k
            
            if time.time() > deadline:
                return True
            
            if len(current_indices) == k:
                # current_sums already verified dissociated by construction
                best_result = [elements[i] for i in current_indices]
                best_k = k
                found = True
                return True
            
            remaining_needed = k - len(current_indices)
            if start_idx + remaining_needed > n:
                return False
            
            for i in range(start_idx, n - remaining_needed + 1):
                v = sorted_vals[i]
                shifted = {s + v for s in current_sums}
                if shifted & current_sums:
                    continue  # Not dissociated, skip
                new_sums = current_sums | shifted
                new_indices = current_indices + [i]
                
                if dfs(i + 1, new_indices, new_sums):
                    return True
                if found:
                    return True
                    
            return False
        
        # Start DFS with empty set
        initial_sums = {0}
        if dfs(0, [], initial_sums):
            break
    
    return best_result
# EVOLVE-BLOCK-END
