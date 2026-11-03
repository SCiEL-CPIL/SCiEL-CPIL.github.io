import heapq
import itertools
import time

def construct(n_rows, k):
    # Baseline: Greedy
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Precompute binomials
    binom = [[0]*(k+2) for _ in range(k+2)]
    for n in range(k+2):
        for r in range(n+1):
            if r == 0 or r == n:
                binom[n][r] = 1
            else:
                binom[n][r] = binom[n-1][r-1] + binom[n-1][r]
                
    total_diffs_per_row = binom[k+1][2]

    def get_heuristic(i, row, used_mask):
        # Calculate number of diffs needed
        diffs_cur_rem = total_diffs_per_row - binom[len(row)][2]
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * total_diffs_per_row
        
        need = diffs_cur_rem + diffs_full_rem
        
        if need == 0:
            return 0
            
        # Find 'need' smallest positive integers not in used_mask
        found = 0
        max_new = 0
        d = 1
        while found < need:
            if not (used_mask & (1 << d)):
                found += 1
                max_new = d
            d += 1
        
        return max_new

    # A* Search
    # State: (f, h, i, row_tuple, used_mask, prev_max)
    # i: index of current row
    # row_tuple: tuple of current row elements
    # used_mask: bitmask of used differences
    # prev_max: max element in all completed rows
    
    start_time = time.time()
    timeout = 115.0
    
    # Initial state
    start_row = (0,)
    start_h = get_heuristic(0, start_row, 0)
    start_g = 0
    start_f = max(start_g, start_h)
    
    open_list = [(start_f, start_h, 0, start_row, 0, 0)]
    visited = set()
    
    # To avoid re-expanding states, we track (i, row_tuple, used_mask, prev_max)
    # But prev_max is derived from path. 
    # Actually, for a given (i, row_tuple, used_mask), prev_max is not unique?
    # No, prev_max is the max of previous rows. The previous rows are determined by the path.
    # Different paths can lead to the same (i, row_tuple, used_mask) but different prev_max?
    # Yes. E.g., Row 1: [0, 1, 2] (max 2). Row 1: [0, 2, 3] (max 3).
    # If both lead to same used_mask and same current row? Unlikely but possible.
    # However, if prev_max is smaller, it's strictly better.
    # So we should keep the state with the minimum prev_max for a given (i, row_tuple, used_mask).
    # Let's use a dictionary: best_prev_max[(i, row_tuple, used_mask)] = min_prev_max
    
    best_prev_max = {}
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, h, i, row, used_mask, prev_max = heapq.heappop(open_list)
        
        state_key = (i, row, used_mask)
        
        # Prune if we have seen this state with a smaller or equal prev_max
        if state_key in best_prev_max:
            if best_prev_max[state_key] <= prev_max:
                continue
        else:
            best_prev_max[state_key] = prev_max
            
        # Calculate current g
        g = max(prev_max, row[-1])
        
        # If we found a complete solution
        if i == n_rows - 1 and len(row) == k + 1:
            # This is a goal state
            # Since A* pops by f, and f=g for goal (h=0), this is the optimal scope
            # But wait, we are searching for scope < best_scope?
            # The heuristic is admissible, so the first goal popped has the minimal possible scope.
            # We can return immediately.
            scope = g
            # Construct the full list of rows? 
            # We didn't store the path in the state to save memory.
            # We need to reconstruct the path or store it.
            # Storing path in state makes it heavy.
            # Alternative: Store parent pointers? 
            # Or, since we just need to return *a* set of rows, and the search is for the *minimal* scope,
            # we can just return the greedy solution if we don't find a better one quickly?
            # No, we must return the optimal one found by A*.
            # To reconstruct, we need the path.
            # Let's add `path` to the state? 
            # That will increase memory significantly.
            # Alternative: Store `prev_row` and `prev_i`?
            # Let's try to store the path as a tuple of tuples in the state.
            # For n_rows=7, k=5, this is manageable.
            
            # REVISION: I will change the state to include the path.
            # State: (f, h, i, row_tuple, used_mask, prev_max, path_tuple)
            # path_tuple: tuple of completed rows.
            
            # This requires updating the heap push logic.
            pass # Placeholder, will rewrite loop below

    # REWRITING THE SEARCH LOOP WITH PATH STORAGE
    
    # Clear previous structures
    best_prev_max = {}
    
    # Initial state with path
    start_path = ()
    open_list = [(start_f, start_h, 0, start_row, 0, 0, start_path)]
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, h, i, row, used_mask, prev_max, path = heapq.heappop(open_list)
        
        state_key = (i, row, used_mask)
        
        if state_key in best_prev_max:
            if best_prev_max[state_key] <= prev_max:
                continue
        else:
            best_prev_max[state_key] = prev_max
            
        g = max(prev_max, row[-1])
        
        # Goal check
        if i == n_rows - 1 and len(row) == k + 1:
            # We found a solution.
            # Since A* is optimal for minimax with admissible h, this is the best.
            full_rows = list(path) + [list(row)]
            return full_rows
        
        # Expand
        if len(row) < k + 1:
            last = row[-1]
            # Try next x
            # Limit x to best_scope - 1 to find improvement
            # If we don't find improvement, we return best_rows (greedy)
            # But we want to find the *optimal* scope.
            # If the optimal scope is equal to greedy, we might not find it if we prune x >= best_scope.
            # We should allow x up to best_scope.
            # If we find a solution with scope == best_scope, it's not an improvement, but it's valid.
            # However, if there's a solution with scope < best_scope, we want it.
            # So we iterate x up to best_scope.
            
            for x in range(last + 1, best_scope + 1):
                new_diffs = [x - a for a in row]
                # Check if any diff is used
                valid = True
                new_mask = 0
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    new_row = row + (x,)
                    new_used = used_mask | new_mask
                    new_prev_max = prev_max
                    new_g = max(new_prev_max, x)
                    
                    # Prune if new_g >= best_scope? 
                    # If new_g == best_scope, we might still complete with scope best_scope.
                    # If we already have a solution with best_scope, we don't need another one with same scope.
                    # But we want to be sure we didn't miss a better one.
                    # If new_g > best_scope, prune.
                    if new_g > best_scope:
                        continue
                        
                    new_h = get_heuristic(i, new_row, new_used)
                    new_f = max(new_g, new_h)
                    
                    # Prune if new_f >= best_scope?
                    # If new_f >= best_scope, it means the lower bound is already >= best_scope.
                    # So we cannot improve on best_scope.
                    if new_f >= best_scope:
                        continue
                        
                    heapq.heappush(open_list, (new_f, new_h, i, new_row, new_used, new_prev_max, path))
        
        elif len(row) == k + 1:
            # Move to next row
            new_i = i + 1
            new_row = (0,)
            new_used = used_mask
            new_prev_max = max(prev_max, row[-1])
            new_path = path + (row,)
            
            new_g = new_prev_max
            
            # Prune if new_g >= best_scope?
            # If new_g >= best_scope, the final scope will be at least new_g.
            # If new_g == best_scope, we can't improve.
            if new_g > best_scope:
                continue
                
            new_h = get_heuristic(new_i, new_row, new_used)
            new_f = max(new_g, new_h)
            
            # Prune if new_f >= best_scope
            if new_f >= best_scope:
                continue
                
            heapq.heappush(open_list, (new_f, new_h, new_i, new_row, new_used, new_prev_max, new_path))

    return best_rows
