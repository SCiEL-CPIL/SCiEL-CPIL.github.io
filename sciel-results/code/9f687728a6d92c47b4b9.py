import random
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def greedy_complete(rows, used_diffs, max_val_limit=None):
        """
        Attempts to complete the rows to full length using a greedy strategy.
        Returns (completed_rows, is_success, max_val).
        """
        rows = [r[:] for r in rows]
        used_diffs = set(used_diffs)
        
        for _ in range(n_rows * row_len): # Max possible steps
            # Find row to extend
            # Heuristic: Row with smallest length, tie-breaker: smallest last value
            target_idx = -1
            min_len = row_len + 1
            min_last = float('inf')
            
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_idx = i
                    min_last = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last:
                        min_last = rows[i][-1]
                        target_idx = i
                        
            if target_idx == -1:
                break
            
            current_row = rows[target_idx]
            last_val = current_row[-1]
            
            # Try to find a valid next value
            # If max_val_limit is set, we must stay below it
            # We try values from last_val + 1 upwards
            found = False
            x = last_val + 1
            while True:
                if max_val_limit is not None and x > max_val_limit:
                    break
                if is_valid_extension(current_row, x, used_diffs):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used_diffs.add(x - v)
                    found = True
                    break
                x += 1
                
            if not found:
                return None, False, 0
                
        # Check if all rows are full
        if all(len(r) == row_len for r in rows):
            max_val = max(max(r) for r in rows)
            return rows, True, max_val
        else:
            return None, False, 0

    def local_search_improve(rows, used_diffs):
        """
        Try to reduce the max value by modifying the rows.
        Strategy: Identify the row with the max value. Try to replace its last few elements
        with smaller valid values.
        """
        best_rows = [r[:] for r in rows]
        best_used = set(used_diffs)
        best_max = max(max(r) for r in best_rows)
        
        # Identify the "bottleneck" row (the one with the max element)
        bottleneck_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
        
        # Try to shorten this row
        # We will try to replace the last element, then the second last, etc.
        # This is a simplified local search.
        
        # For this implementation, we'll try a random restart of the last few elements
        # But to keep it simple and fast, we'll just try to find a smaller valid last element
        # for the bottleneck row, assuming the rest is fixed.
        
        row = best_rows[bottleneck_idx]
        if len(row) < row_len:
             return best_rows, best_used, best_max # Can't improve if not full? 
             # Actually, local search should work on full rows.
             
        # Remove last element
        last_val = row.pop()
        # Remove its diffs from used
        for v in row:
            best_used.remove(last_val - v)
            
        # Try to find a new smaller last element
        # We need a value > row[-1] and < last_val
        new_val = row[-1] + 1
        improved = False
        while new_val < last_val:
            if is_valid_extension(row, new_val, best_used):
                # Check if this keeps the row increasing (it does by construction)
                # But we also need to ensure we don't break other constraints?
                # The diffs are distinct globally. We removed old ones, adding new ones.
                # If new_val < last_val, the max value decreases.
                row.append(new_val)
                for v in row[:-1]:
                    best_used.add(new_val - v)
                improved = True
                best_max = max(max(r) for r in best_rows)
                break
            new_val += 1
            
        if not improved:
            # Restore original
            row.append(last_val)
            for v in row[:-1]:
                best_used.add(last_val - v)
                
        return best_rows, best_used, best_max

    start_time = time.time()
    timeout = 110
    
    # 1. Initial Greedy Solution
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    
    best_rows, success, best_cost = greedy_complete(initial_rows, initial_used)
    if not success:
        # Fallback: simple greedy without limit
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            i = min(range(n_rows), key=lambda idx: (len(rows[idx]), rows[idx][-1]))
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - v for v in rows[i]}
                if not (new_diffs & used):
                    rows[i].append(x)
                    used.update(new_diffs)
                    break
                x += 1
        best_rows = rows
        best_cost = max(max(r) for r in best_rows)
        best_used = get_used_diffs(best_rows)

    # 2. Iterated Local Search
    # We will try to improve the solution by randomizing the construction slightly
    # and then completing it, or by local moves.
    
    num_restarts = 50 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Perturb the best solution
        # Strategy: Take the best_rows, remove the last element of a random row,
        # and re-complete that row with a different greedy choice.
        
        candidate_rows = [r[:] for r in best_rows]
        candidate_used = set(best_used)
        
        # Pick a random row to perturb
        if n_rows > 0:
            target_row_idx = random.randint(0, n_rows - 1)
            row = candidate_rows[target_row_idx]
            
            # Remove last element
            if len(row) > 1:
                last_val = row.pop()
                for v in row:
                    candidate_used.discard(last_val - v)
                    
                # Re-complete this row with a randomized greedy
                # We try to pick values that are slightly different from the original
                # to explore the neighborhood
                
                # Simple randomization: try a few candidates
                # Original last val was L. We want to find a valid L' < L or just different.
                # But we need to complete the row to full length.
                
                # Let's just re-greedy this row from its current state
                # with a slight bias towards smaller values if possible
                
                # To keep it simple, we'll just use the standard greedy completion
                # but we've already removed the last element, so it might pick the same one.
                # To force a change, we can ban the previous value.
                
                banned_val = last_val
                x = row[-1] + 1
                while True:
                    if x == banned_val:
                        x += 1
                        continue
                    if is_valid_extension(row, x, candidate_used):
                        row.append(x)
                        for v in row[:-1]:
                            candidate_used.add(x - v)
                        break
                    x += 1
                    
                # Now complete the rest of the rows (if any were affected? No, only this row changed)
                # But wait, we only removed 1 element. The row might not be full yet if we removed more?
                # We only removed 1. So the row is now length row_len - 1.
                # We added 1. So it is full again.
                # However, the *other* rows are unchanged.
                # So we just need to check if this new configuration is valid and better.
                
                # Check validity (should be valid by construction)
                # Calculate new cost
                new_cost = max(max(r) for r in candidate_rows)
                
                if new_cost < best_cost:
                    best_rows = candidate_rows
                    best_cost = new_cost
                    best_used = candidate_used

    return best_rows
