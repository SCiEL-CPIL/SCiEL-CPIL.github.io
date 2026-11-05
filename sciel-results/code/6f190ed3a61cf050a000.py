import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set (generalized Golomb rulers) 
    minimizing the largest entry.
    """
    # Deadline: 100 seconds to leave buffer for Python overhead and return
    deadline = time.time() + 100.0
    
    # 1. Greedy Baseline
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check conflicts
                conflict = False
                # Calculate mask for new diffs
                new_mask = 0
                for a in row:
                    d = x - a
                    new_mask |= (1 << d)
                if new_mask & used:
                    conflict = True
                if not conflict:
                    row.append(x)
                    used |= new_mask
            rows.append(row)
        return rows

    greedy_sol = greedy()
    best_sol = [list(r) for r in greedy_sol]
    best_max = max(max(r) for r in greedy_sol)
    
    # If k is very small or n_rows is 1, greedy might be optimal or close.
    # But let's try to improve.
    
    # 2. DFS with Backtracking and Bitset Optimization
    # We search for a solution with max < best_max.
    # We update best_max as we find better solutions.
    
    # State for DFS
    rows = []
    current_row = []
    used_mask = 0
    
    # We need to pass these around or use nonlocal/global in nested function
    # Using a class or nonlocal variables in a nested function is cleaner.
    
    def dfs(row_idx, pos):
        nonlocal best_max, best_sol, used_mask, current_row, rows
        
        # Time check
        if time.time() > deadline:
            return False
        
        # Pruning: If we are filling a row, and the last element added (current max of this row) 
        # is already >= best_max, we cannot improve the global max.
        # Note: If pos > 0, current_row[-1] is the max of the current row so far.
        # If pos == 0, current_row is [0], max is 0.
        if pos > 0 and current_row[-1] >= best_max:
            return False
        
        # Base case: All rows completed
        if row_idx == n_rows:
            # We have a valid solution.
            # Calculate max of this solution
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_sol = [list(r) for r in rows]
            return True
        
        # If we are starting a new row
        if pos == 0:
            # Start new row with 0
            current_row = [0]
            # Recurse to next pos in this row
            return dfs(row_idx, 1)
        
        # If we are continuing a row
        if pos == k + 1:
            # Row is complete
            rows.append(list(current_row))
            # Start next row
            res = dfs(row_idx + 1, 0)
            rows.pop()
            return res
        
        # We need to find the next element x for current_row
        # x must be > current_row[-1]
        last_x = current_row[-1]
        
        # Iterate through possible x
        # To be efficient, we start from last_x + 1
        # We can bound the upper limit by best_max - 1 (since we want to improve)
        # Actually, if we pick x such that x >= best_max, it won't improve the global max 
        # IF this row determines the max. But other rows might be smaller.
        # However, if any element is >= best_max, the global max is >= best_max.
        # So we only care about x < best_max.
        
        for x in range(last_x + 1, best_max):
            if time.time() > deadline:
                return False
            
            # Check conflicts
            new_mask = 0
            for a in current_row:
                d = x - a
                new_mask |= (1 << d)
            
            if new_mask & used_mask:
                continue
            
            # Valid candidate
            current_row.append(x)
            used_mask |= new_mask
            
            # Recurse
            if dfs(row_idx, pos + 1):
                # If a better solution was found, we could return True to stop?
                # No, we want the BEST. But the pruning `current_row[-1] >= best_max` 
                # will cut off branches that can't beat the new best.
                # So we can continue searching to see if we find something even better?
                # Actually, if we found a solution with max M, any branch with max >= M is pruned.
                # So the search naturally focuses on max < M.
                # We can return True to unwind, but the top level will keep going if we structure it that way?
                # Let's just return True to indicate "solution found". 
                # The pruning handles the optimization.
                pass 
            else:
                # Backtrack
                current_row.pop()
                used_mask ^= new_mask
                continue
                
            # If we returned True from recursion, we can try to backtrack to find an even better one?
            # The pruning ensures that only branches with potential to beat `best_max` are explored.
            # So we should continue the loop to try other x values?
            # If we found a solution, `best_max` is updated.
            # The next `x` in the loop will be larger, so `current_row[-1]` will be larger.
            # If `x` was close to `best_max`, the new `best_max` might be smaller than the next `x`.
            # Then `current_row[-1] >= best_max` check will prune.
            # So it's safe to continue.
            
            # Backtrack after successful recursion?
            # If we found a solution, we still need to backtrack to try other options 
            # that might lead to an even smaller max?
            # Yes.
            current_row.pop()
            used_mask ^= new_mask
            
        return False

    # Start DFS
    dfs(0, 0)
    
    return best_sol

