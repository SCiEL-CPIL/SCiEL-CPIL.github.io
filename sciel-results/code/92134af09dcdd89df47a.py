import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_scope = float('inf')
    best_rows = None
    
    # Parameters for randomization
    # We will run multiple restarts with different random seeds
    
    for restart in range(10000): # Upper bound on restarts, but we check time
        if time.time() > deadline:
            break
        
        # Randomize the search
        # We can randomize the order in which we try candidates or the initial "bias"
        
        rows = []
        used_diffs = set()
        
        # Backtracking function
        # State: current_row_idx, current_pos_in_row, current_row, used_diffs
        # We want to minimize the max element.
        
        # To speed up, we can use a heuristic: try smallest valid x first, 
        # but with some randomization to escape local minima.
        
        def backtrack(row_idx, pos, row, used):
            nonlocal best_scope, best_rows
            
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                scope = max(rows[-1][-1]) if rows else 0
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                return True
            
            if pos == k + 1:
                # Row complete, move to next row
                rows.append(row[:])
                return backtrack(row_idx + 1, 0, [0], used)
            
            # Determine the range for x
            # x must be > row[-1]
            # Also, to avoid infinite loops or huge numbers, we can set a limit
            # The theoretical max is related to the counting bound.
            # For (7,5), bound is 105. Best known 112.
            # Let's set a dynamic upper bound for x.
            # If we are in the first row, we can try up to some limit.
            # If we are in later rows, the limit can be tighter if we are already large.
            
            # Heuristic: The next x should not be too large.
            # Let's try candidates from row[-1] + 1 upwards.
            # To allow backtracking, we need to try multiple candidates.
            
            # Optimization: Only try candidates that don't immediately conflict?
            # The check is O(k).
            
            # Let's define a limit for x to prune.
            # If we are already worse than best_scope, prune.
            # But we don't know the final scope yet.
            
            # Simple limit: 
            # The sum of differences is bounded.
            # Let's just try up to best_scope - 1 or a fixed reasonable limit if best_scope is inf.
            # For (7,5), 112 is the target. Let's use 150 as a hard limit for x in early stages?
            # Actually, if we exceed best_scope, we can prune.
            
            current_max = row[-1]
            
            # If we have a best_scope, we can prune if current_max + (k+1-pos) > best_scope?
            # Not strictly, but if current_max is already >= best_scope, we can prune.
            
            if current_max >= best_scope:
                return False
            
            # Generate candidates
            # We want to try small x first, but randomize slightly.
            # Let's try candidates in a randomized order within a window, or just smallest first with random restarts.
            # The insight says "randomized greedy with window search".
            # This implies we might pick from a window of valid candidates.
            
            # Let's collect valid candidates in a window [current_max + 1, current_max + window_size]
            # And then pick one randomly? Or try them in order?
            # If we try them in order, it's deterministic greedy.
            # If we pick randomly, we explore more.
            
            # Let's try a hybrid: Try the smallest valid one, but if it fails, backtrack.
            # To add randomization, we can shuffle the candidates in a window.
            
            window_size = 10 # Tune this
            candidates = []
            upper_bound = current_max + window_size
            # Also cap by best_scope - 1
            if best_scope < float('inf'):
                upper_bound = min(upper_bound, best_scope - 1)
                
            for x in range(current_max + 1, upper_bound + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
            
            # If no candidates in window, we might need to look further?
            # Or just return False (backtrack).
            # If we return False, the parent will try a different x.
            
            # Randomize the order of candidates to escape local minima
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                # Re-check just in case (though it was checked above)
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    
                    if backtrack(row_idx, pos + 1, row, used):
                        return True
                    
                    # Undo
                    used.difference_update(new_diffs)
                    row.pop()
            
            return False
        
        # Start the search
        # We can also randomize the start of the first row? No, it must start at 0.
        # But we can randomize the "window" or the shuffle.
        
        # To diversify, we can also randomize the `window_size` or the shuffle seed.
        # The loop `for restart` handles restarts.
        
        # Initial state
        row = [0]
        used = set()
        
        # We need to pass `rows` and `used` correctly.
        # My backtrack function modifies `rows` and `used` in place?
        # `rows` is appended to when a row is complete.
        # `used` is updated/undone.
        
        # Wait, `rows` is a list of completed rows.
        # In `backtrack`, when `pos == k+1`, we append `row` to `rows`.
        # This modifies the outer `rows` list.
        # We need to be careful with backtracking `rows`.
        # If we backtrack from a later row, we need to remove the row we added.
        
        # Let's refactor `backtrack` to handle `rows` properly or pass it as a local state.
        
        def backtrack_v2(row_idx, pos, row, used, rows_state):
            nonlocal best_scope, best_rows
            
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                scope = max(rows_state[-1][-1]) if rows_state else 0
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows_state]
                return True
            
            if pos == k + 1:
                rows_state.append(row[:])
                if backtrack_v2(row_idx + 1, 0, [0], used, rows_state):
                    return True
                rows_state.pop()
                return False
            
            current_max = row[-1]
            
            if current_max >= best_scope:
                return False
            
            window_size = 15
            candidates = []
            upper_bound = current_max + window_size
            if best_scope < float('inf'):
                upper_bound = min(upper_bound, best_scope - 1)
                
            for x in range(current_max + 1, upper_bound + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
            
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    
                    if backtrack_v2(row_idx, pos + 1, row, used, rows_state):
                        return True
                    
                    used.difference_update(new_diffs)
                    row.pop()
            
            return False
        
        rows_state = []
        backtrack_v2(0, 0, [0], set(), rows_state)
        
    if best_rows is None:
        # Fallback to greedy if search failed (should not happen for small inputs)
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
        
    return best_rows
