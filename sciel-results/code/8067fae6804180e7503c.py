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
    
    best_result = [1]  # fallback
    best_k = 1
    
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
    
    def _greedy_build(elements, sorted_vals):
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
    
    def _dfs_search(elements, sorted_vals, target_k, deadline):
        """DFS to find a dissociated subset of size target_k.
        Returns list of element indices or None."""
        found = [None]
        
        def dfs(start_idx, current_indices, current_sums):
            if time.time() > deadline:
                return True
            
            cur_len = len(current_indices)
            if cur_len == target_k:
                found[0] = current_indices[:]
                return True
            
            remaining_needed = target_k - cur_len
            if start_idx + remaining_needed > n:
                return False
            
            for i in range(start_idx, n - remaining_needed + 1):
                if time.time() > deadline:
                    return True
                v = sorted_vals[i]
                shifted = {s + v for s in current_sums}
                if shifted & current_sums:
                    continue  # Not dissociated, skip
                new_sums = current_sums | shifted
                new_indices = current_indices + [i]
                
                if dfs(i + 1, new_indices, new_sums):
                    return True
                if found[0] is not None:
                    return True
                    
            return False
        
        dfs(0, [], {0})
        return found[0]
    
    # Try multiple orderings to find the largest dissociated set
    import random
    rng = random.Random(42)
    
    # Define several orderings to try
    orderings = []
    
    # 1. Descending by reciprocal value (original)
    orderings.append(sorted(range(n), key=lambda i: -vals[i+1]))
    
    # 2. Ascending by reciprocal value
    orderings.append(sorted(range(n), key=lambda i: vals[i+1]))
    
    # 3. By x value ascending (1, 2, 3, ...)
    orderings.append(list(range(n)))
    
    # 4. By x value descending (22, 21, ..., 1)
    orderings.append(list(range(n-1, -1, -1)))
    
    # 5-10. Random shuffles
    for _ in range(6):
        orderings.append(rng.sample(range(n), n))
    
    # Try each ordering
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        elements = [ordering[i] + 1 for i in range(n)]
        sorted_vals = [vals[ordering[i] + 1] for i in range(n)]
        
        # Greedy first
        greedy_result = _greedy_build(elements, sorted_vals)
        if len(greedy_result) > best_k:
            best_k = len(greedy_result)
            best_result = [elements[i] for i in greedy_result]
        
        # DFS for larger sets
        for k in range(n, best_k, -1):
            if time.time() > deadline:
                break
            
            result = _dfs_search(elements, sorted_vals, k, deadline)
            if result is not None:
                best_k = k
                best_result = [elements[i] for i in result]
                break
    
    return best_result
# EVOLVE-BLOCK-END
