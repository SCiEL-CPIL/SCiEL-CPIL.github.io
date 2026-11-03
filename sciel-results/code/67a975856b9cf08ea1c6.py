import time
import random

def construct(n_rows, k):
    # Fallback greedy
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    start_time = time.time()
    time_limit = 115.0
    
    diffs_per_row = (k + 1) * k // 2
    lb = n_rows * diffs_per_row
    
    # We will try M from lb upwards
    # Using bitmask for 'used' differences is much faster
    # Max M we expect to try is around 125, so bitmask is fine.
    
    for M in range(lb, lb + 50):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Set a sub-time limit for this M to avoid getting stuck
        # If M is small, search space is smaller, but constraints are tighter.
        # If M is large, search space is huge.
        # Let's allow 10-20 seconds per M, decreasing as we go higher?
        # Or just a fixed budget.
        
        used_mask = 0
        rows = []
        
        # Heuristic: We can randomize the choice of x to explore different branches
        # But for a specific M, we want to find *any* solution.
        # Deterministic DFS is fine if we have a good ordering.
        # Ordering: Try x that creates "common" differences first? Or "rare"?
        # Actually, just trying small x first is a good heuristic for minimization.
        
        def dfs(row_idx, curr_row):
            nonlocal used_mask, rows, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
            
            if len(curr_row) == k + 1:
                if row_idx == n_rows - 1:
                    rows.append(list(curr_row))
                    return True
                return dfs(row_idx + 1, [0])
            
            last = curr_row[-1]
            start_val = last + 1
            
            # Calculate mask for new differences
            # We need to check if any new difference conflicts with used_mask
            
            # Optimization: Precompute differences for candidate x?
            # For each x in range(start_val, M+1):
            #    diffs = [x - a for a in curr_row]
            #    mask = sum(1 << d for d in diffs)
            #    if used_mask & mask: continue
            #    ...
            
            # To add randomization/restart capability, we can shuffle the candidates
            # but only if we have failed deterministic search?
            # Let's stick to deterministic first, but maybe swap order if stuck?
            # Actually, let's just do a standard DFS.
            
            for x in range(start_val, M + 1):
                # Calculate bitmask for differences
                mask = 0
                conflict = False
                for a in curr_row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    mask |= bit
                
                if conflict:
                    continue
                
                # Check if we can fit the remaining elements?
                # Simple bound: We need (k+1 - len(curr_row) - 1) more elements.
                # The smallest next is x+1.
                # The largest is M.
                # This bound is weak.
                
                used_mask |= mask
                curr_row.append(x)
                
                if dfs(row_idx, curr_row):
                    return True
                
                curr_row.pop()
                used_mask ^= mask # Remove mask (since no bits were added by others, XOR is same as subtract if disjoint, but OR is safer if we track properly. Actually, since we check conflict, mask is disjoint from used_mask BEFORE update. So used_mask_after = used_mask_before | mask. To revert, used_mask_before = used_mask_after & ~mask. Or just store old mask.)
                # Safer:
                # old_mask = used_mask
                # used_mask |= mask
                # ...
                # used_mask = old_mask
                # Let's use the restore method to be safe.
                
            return False
        
        # To make it faster, we can use a local variable for used_mask and restore it
        # The `nonlocal` and XOR/AND approach is tricky if multiple branches add bits.
        # Let's use a stack of masks or just pass it as argument.
        
        def dfs_v2(row_idx, curr_row, used_mask):
            if time.time() - start_time > time_limit:
                return False
            
            if len(curr_row) == k + 1:
                if row_idx == n_rows - 1:
                    rows.append(list(curr_row))
                    return True
                return dfs_v2(row_idx + 1, [0], used_mask)
            
            last = curr_row[-1]
            start_val = last + 1
            
            # Heuristic: Try values that are "sparse"?
            # Just iterate.
            for x in range(start_val, M + 1):
                mask = 0
                conflict = False
                for a in curr_row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    mask |= bit
                
                if conflict:
                    continue
                
                curr_row.append(x)
                if dfs_v2(row_idx, curr_row, used_mask | mask):
                    return True
                curr_row.pop()
                
            return False

        if dfs_v2(0, [0], 0):
            return rows

    return greedy_construct()
