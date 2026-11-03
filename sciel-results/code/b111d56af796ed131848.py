import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0 # Safety margin for 120s limit
    row_len = k + 1
    
    # Bitmask for differences. Max difference is bounded by scope.
    # Scope is likely < 200 for 7x6. 1 << 200 is huge but Python handles big ints.
    # We can use a set of integers for differences if scope is large, but bitmask is faster for small scopes.
    # Let's use a set for flexibility and simplicity, or a bitmask if we know the upper bound.
    # Given the bound is ~112, a bitmask of size 200 is fine.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def find_min_row(used_diffs, max_scope_limit, time_limit):
        """
        Finds a row of length row_len starting with 0, strictly increasing,
        with no differences in used_diffs, minimizing the last element.
        Returns (row, diffs_set) or (None, None) if not found within limit.
        """
        best_row = [None]
        best_last = [max_scope_limit + 1]
        start_time = time.time()
        
        # DFS parameters
        # row: current partial row
        # last_val: last element in row
        # curr_diffs: set of differences in current partial row
        
        def dfs(row, last_val, curr_diffs):
            if time.time() > time_limit:
                return
            if len(row) == row_len:
                if last_val < best_last[0]:
                    best_last[0] = last_val
                    best_row[0] = row[:]
                return
            
            # Pruning: if last_val is already >= best_last[0], no point continuing
            if last_val >= best_last[0]:
                return
                
            # Try next values
            # Heuristic: start from last_val + 1
            # We can also cap the search if last_val grows too large
            max_next = min(best_last[0] - 1, max_scope_limit)
            
            for x in range(last_val + 1, max_next + 1):
                # Check conflicts with used_diffs and curr_diffs
                # New diffs are x - a for a in row
                conflict = False
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_diffs or d in curr_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                    
                # Recursive step
                row.append(x)
                curr_diffs.update(new_diffs)
                
                dfs(row, x, curr_diffs)
                
                row.pop()
                curr_diffs.difference_update(new_diffs)
                
                if time.time() > time_limit:
                    return
                
                # If we found a solution with last_val == last_val + 1 (minimal increase), 
                # we might be able to prune more aggressively, but let's stick to standard DFS.
                # Actually, if we find a solution, best_last is updated. 
                # The loop continues to see if a smaller last_val is possible? 
                # No, x is increasing. If we found a solution at x, any larger x will have larger last_val.
                # So we can break?
                # Wait, the "last_val" of the final row is the x at the last step.
                # If we are at step j, and we pick x, the final last_val will be >= x.
                # If we found a complete row with last_val = L, and we are currently exploring x > L, 
                # we can prune. The check `if last_val >= best_last[0]: return` handles this at the start of dfs.
                # Inside the loop, if we just found a solution, best_last[0] is updated.
                # The next x in the loop will be larger.
                # So we can break the loop if we found a solution? 
                # Yes, because x is increasing, so any subsequent x will lead to a row with last element >= current x > best_last[0] (if best_last was updated to current x).
                # Actually, best_last is the last element of the *complete* row.
                # If we are at depth d, and we pick x, the final last element will be >= x.
                # If we found a complete row with last element L, and we are at depth d < row_len, 
                # and we pick x such that x > L, then the final last element will be > L.
                # So if we found a solution, we can stop searching for larger x at this level?
                # Not necessarily, because a smaller x might have led to a dead end, and a larger x might lead to a valid row with a *smaller* final last element? No, that's impossible.
                # If x1 < x2, and both lead to valid rows, the row from x1 will have last element <= last element of row from x2?
                # Not strictly guaranteed if the path diverges, but generally, smaller early choices allow smaller later choices.
                # However, standard DFS with `best_last` pruning is sufficient.
                
                if best_row[0] is not None:
                    # If we found a solution, we can break if the current x is already >= best_last[0]?
                    # The check at the top of dfs handles this for deeper levels.
                    # For the current level, if we found a solution, best_last[0] is the last element of that solution.
                    # The next x is x+1. If x+1 > best_last[0], we can break.
                    if x + 1 > best_last[0]:
                        break

        dfs([0], 0, set())
        
        if best_row[0] is None:
            return None, None
        return best_row[0], get_diffs(best_row[0])

    # Main construction loop
    rows = []
    used_diffs = set()
    
    # We can try multiple random orderings or restarts if the first one fails or is suboptimal.
    # But for 7x6, the greedy sequential approach usually works.
    # Let's add a time check.
    
    # Estimate max scope. Start with a generous bound.
    # Counting bound is 105. Best known 112. Let's try 150.
    
    for i in range(n_rows):
        remaining_time = DEADLINE - time.time()
        if remaining_time < 1.0:
            break
        
        # Allocate time for this row search
        # Earlier rows might need more time if they are "tighter"?
        # Let's give each row a chunk of the remaining time.
        row_time_limit = min(remaining_time - 0.1, 5.0) # Cap at 5s per row to allow retries
        
        row, diffs = find_min_row(used_diffs, 200, row_time_limit)
        
        if row is None:
            # If DFS fails to find a row with scope < 200, try a greedy fallback
            # This is unlikely for 7x6, but good for safety.
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                conflict = False
                for a in row:
                    if x - a in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    # Also check against current row diffs
                    local_diffs = set()
                    for a in row:
                        local_diffs.add(x - a)
                    if local_diffs & set(used_diffs):
                        conflict = True
                    if not conflict:
                        row.append(x)
                        used_diffs.update(local_diffs)
                        break
            # Re-calculate diffs for the greedy row
            diffs = get_diffs(row)
        else:
            used_diffs.update(diffs)
            
        rows.append(row)
        
    # If we ran out of time or failed, we might have fewer rows.
    # Ensure we have n_rows rows. If not, pad with greedy.
    while len(rows) < n_rows:
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            conflict = False
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            if conflict:
                continue
            # Check local conflicts
            if new_diffs & used_diffs:
                conflict = True
            else:
                # Check against row's own previous diffs? 
                # The loop above checks against `row` which is the partial row.
                # But we need to ensure no conflict with *existing* diffs in `row`.
                # The `new_diffs` are diffs with elements in `row`.
                # We need to ensure these `new_diffs` don't conflict with each other? 
                # No, they are all differences from x to a. They are distinct if a are distinct.
                # We need to ensure they don't conflict with `used_diffs`.
                # And they don't conflict with diffs *within* the row? 
                # The row is built incrementally. The diffs within the row are generated as we go.
                # The `used_diffs` set contains all diffs from *previous* rows.
                # We need to ensure the new row's diffs don't conflict with `used_diffs`.
                # The check `if d in used_diffs` handles that.
                # Do we need to check if `new_diffs` have internal conflicts? 
                # No, internal conflicts in a row are allowed? 
                # "all differences between two entries of the same row are distinct over all rows"
                # This means: 
                # 1. Within a row, differences are distinct? 
                # "distinct over all rows" usually means the set of all differences from all rows is a set (no duplicates).
                # So yes, all differences (within any row) must be unique globally.
                # So if a row has a difference d, no other row can have d, and the same row cannot have d twice.
                # Since the row is strictly increasing, differences within a row are naturally distinct 
                # UNLESS the row has repeated differences? 
                # E.g., row [0, 1, 3]. Diffs: 1, 3, 2. All distinct.
                # Row [0, 1, 2]. Diffs: 1, 2, 1. Duplicate 1! 
                # So we must ensure that within the row being constructed, the new difference `x - a` 
                # is not equal to any difference already in the row.
                
                # My `find_min_row` checks `d in curr_diffs`. 
                # My greedy fallback above does NOT check `d` against previous diffs in the *same* row.
                # Let's fix the greedy fallback.
                
                pass # Placeholder, will rewrite greedy properly below
        
        # Proper Greedy Fallback
        row = [0]
        x = 0
        row_diffs = set()
        while len(row) < row_len:
            x += 1
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs or d in row_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                row_diffs.update(new_diffs)
                used_diffs.update(new_diffs)
                break # This break is wrong, it's inside the x loop? No, x is incremented in while.
                # The while loop condition is len(row) < row_len.
                # So we add x, update, and continue the while loop (next x).
                # Wait, the `break` above would exit the `while len(row)...` loop? No, it's inside `for a in row`.
                # Let's restructure.
        
        # Correct Greedy
        row = [0]
        x = 0
        row_diffs = set()
        while len(row) < row_len:
            x += 1
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs or d in row_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                row_diffs.update(new_diffs)
                used_diffs.update(new_diffs)
        
        rows.append(row)

    # Final check and return
    # Calculate scope
    scope = max(r[-1] for r in rows)
    
    # Optional: Quick local search to improve scope if time allows?
    # Given the tight time, and that the DFS is quite strong, we might not need it.
    # But if we have > 10s left, we could try to swap rows or optimize the last row.
    # For now, return the constructed rows.
    
    return rows

# Wait, the greedy fallback logic in the draft was messy. 
# The main `find_min_row` is the robust part. 
# The greedy is just a fallback. 
# I will simplify the fallback to a simple greedy that ensures validity.

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def find_min_row(used_diffs, max_scope_limit, time_limit):
        best_row = [None]
        best_last = [max_scope_limit + 1]
        start_time = time.time()
        
        def dfs(row, last_val, curr_diffs):
            if time.time() > time_limit:
                return
            if len(row) == row_len:
                if last_val < best_last[0]:
                    best_last[0] = last_val
                    best_row[0] = row[:]
                return
            
            if last_val >= best_last[0]:
                return
                
            max_next = min(best_last[0] - 1, max_scope_limit)
            
            for x in range(last_val + 1, max_next + 1):
                conflict = False
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_diffs or d in curr_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                    
                row.append(x)
                curr_diffs.update(new_diffs)
                
                dfs(row, x, curr_diffs)
                
                row.pop()
                curr_diffs.difference_update(new_diffs)
                
                if time.time() > time_limit:
                    return
                
                if best_row[0] is not None and x + 1 > best_last[0]:
                    break

        dfs([0], 0, set())
        
        if best_row[0] is None:
            return None, None
        return best_row[0], get_diffs(best_row[0])

    def greedy_row(used_diffs):
        row = [0]
        x = 0
        row_diffs = set()
        while len(row) < row_len:
            x += 1
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs or d in row_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                row_diffs.update(new_diffs)
                used_diffs.update(new_diffs)
        return row

    rows = []
    used_diffs = set()
    
    for i in range(n_rows):
        remaining_time = DEADLINE - time.time()
        if remaining_time < 0.5:
            # Not enough time, use greedy
            row = greedy_row(used_diffs)
            rows.append(row)
            continue
            
        row_time_limit = min(remaining_time - 0.5, 5.0)
        
        row, diffs = find_min_row(used_diffs, 200, row_time_limit)
        
        if row is None:
            row = greedy_row(used_diffs)
        else:
            used_diffs.update(diffs)
            
        rows.append(row)

    return rows
