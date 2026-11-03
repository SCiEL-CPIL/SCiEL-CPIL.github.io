"""Difference triangle set construction using constraint satisfaction with backtracking."""
import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_scope = None
    best_rows = None
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Helper: compute all differences for a set of rows
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if rows are valid
    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == total_diffs
    
    # Helper: compute scope
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Greedy construction with randomized ordering
    def greedy_construct(used_diffs, k, max_val, random_order=False):
        """Build one row greedily, avoiding used differences."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            row.append(x)
            used_diffs |= new_diffs
        return row
    
    # Backtracking with forward checking
    def backtrack_search(rows, row_idx, used_diffs, max_scope, depth_limit=50000):
        """
        Try to build remaining rows using backtracking with forward checking.
        Returns (success, rows, used_diffs) or (False, None, None).
        """
        if time.time() > deadline:
            return False, None, None
        
        if row_idx == n_rows:
            return True, [list(r) for r in rows], used_diffs
        
        # Try different starting positions for this row
        # Use least-constraining-value heuristic: try values that leave more options
        candidates = []
        for start_val in range(1, max_scope + 1):
            # Quick check: can we even build this row starting from 0?
            # We need k+1 values, so we need at least k more values after 0
            if start_val > max_scope:
                break
            candidates.append(start_val)
        
        # Shuffle candidates for diversity
        random.shuffle(candidates)
        
        for start_val in candidates:
            if time.time() > deadline:
                return False, None, None
            
            # Build this row greedily with the constraint that all values <= max_scope
            new_row = [0]
            used_new_diffs = set()
            x = 0
            success = True
            while len(new_row) < k + 1:
                x += 1
                if x > max_scope:
                    success = False
                    break
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_diffs or new_diffs & used_new_diffs:
                    continue
                new_row.append(x)
                used_new_diffs |= new_diffs
            
            if not success:
                continue
            
            # Forward checking: verify that we can still build remaining rows
            # Simple heuristic: check if there are enough unused differences
            remaining_rows = n_rows - row_idx - 1
            diffs_needed = remaining_rows * k * (k + 1) // 2
            available_diffs = set(range(1, max_scope + 1)) - (used_diffs | used_new_diffs)
            if len(available_diffs) < diffs_needed:
                continue
            
            # Try this row
            rows.append(new_row)
            new_used_diffs = used_diffs | used_new_diffs
            
            success, result_rows, result_diffs = backtrack_search(
                rows, row_idx + 1, new_used_diffs, max_scope, depth_limit
            )
            
            rows.pop()
            
            if success:
                return True, result_rows, result_diffs
        
        return False, None, None
    
    # Iterative deepening: try increasing max_scope values
    # Start from a reasonable lower bound
    # Counting bound: need total_diffs distinct values in [1, max_scope]
    # So max_scope >= total_diffs
    lower_bound = total_diffs
    
    # Try to find a solution with increasing max_scope
    current_max_scope = lower_bound
    
    while time.time() < deadline:
        # Try backtracking search with current max_scope
        initial_rows = []
        success, result_rows, _ = backtrack_search(
            initial_rows, 0, set(), current_max_scope
        )
        
        if success:
            scope = get_scope(result_rows)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = result_rows
            
            # If we found a solution, try to improve by reducing max_scope
            # But first, let's try a few more with randomization
            break
        
        # If no solution, increase max_scope
        current_max_scope += 5
    
    # If backtracking didn't find anything good, fall back to greedy + local search
    if best_rows is None or best_scope > 150:
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
        
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
    
    # Local search to improve the solution
    if best_rows is not None and time.time() < deadline - 10:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        
        # Try to compact: for each row, try to reduce its max value
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            
            # Pick a random row and try to rebuild it with a lower max
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(current_rows[row_idx])
            
            # Remove this row's differences
            all_diffs = set()
            for i, row in enumerate(current_rows):
                if i != row_idx:
                    for a in range(len(row)):
                        for b in range(a+1, len(row)):
                            all_diffs.add(row[b] - row[a])
            
            # Try to build a new row with max value < old_row[-1]
            target_max = old_row[-1] - 1
            if target_max < k:
                continue
            
            new_row, x = [0], 0
            while len(new_row) < k + 1:
                x += 1
                if x > target_max:
                    break
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & all_diffs):
                    new_row.append(x)
                    all_diffs |= new_diffs
            
            if len(new_row) == k + 1:
                current_rows[row_idx] = new_row
                if is_valid(current_rows):
                    new_scope = get_scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
        
        # Try element-wise modifications
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            row_idx = random.randint(0, n_rows - 1)
            if len(current_rows[row_idx]) <= 1:
                continue
            
            pos = random.randint(1, len(current_rows[row_idx]) - 1)
            old_val = current_rows[row_idx][pos]
            
            for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 20, -20]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= current_rows[row_idx][pos-1]:
                    continue
                if pos < len(current_rows[row_idx]) - 1 and new_val >= current_rows[row_idx][pos+1]:
                    continue
                
                current_rows[row_idx][pos] = new_val
                
                if is_valid(current_rows):
                    new_scope = get_scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                    break
                else:
                    current_rows[row_idx][pos] = old_val
                    break
    
    return best_rows
