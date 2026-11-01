import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Greedy Initialization
    def greedy_init():
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_init()
    
    # Helper to get all diffs for a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Helper to check if a new position for row[r][idx] is valid
    # old_val is removed, new_val is inserted
    def can_move(r, idx, new_val):
        row = rows[r]
        # Bounds check
        if idx > 0 and new_val <= row[idx-1]:
            return False
        if idx < len(row)-1 and new_val >= row[idx+1]:
            return False
            
        # Calculate diffs that will be removed
        old_val = row[idx]
        removed = set()
        for i in range(len(row)):
            if i != idx:
                d = abs(row[i] - old_val)
                removed.add(d)
                
        # Calculate diffs that will be added
        added = set()
        for i in range(len(row)):
            if i != idx:
                d = abs(row[i] - new_val)
                added.add(d)
                
        # Check conflicts with other rows
        # Diffs in 'added' must not be in 'used_diffs' minus 'removed'
        # Actually, 'removed' diffs are currently in 'used_diffs'.
        # So, any diff in 'added' that is NOT in 'removed' must not be in 'used_diffs'.
        # Also, diffs within 'added' must be unique? No, the row is strictly increasing,
        # so diffs from new_val to others are distinct if others are distinct.
        # But we must ensure no collision with other rows.
        
        # Efficient check:
        # For d in added:
        #   if d in used_diffs and d not in removed:
        #       return False
        # Also, we must ensure that the 'removed' set doesn't contain duplicates?
        # No, row elements are distinct, so diffs from one element to others are distinct.
        
        for d in added:
            if d in used_diffs:
                if d not in removed:
                    return False
        return True

    def do_move(r, idx, new_val):
        row = rows[r]
        old_val = row[idx]
        
        # Update used_diffs
        removed = set()
        for i in range(len(row)):
            if i != idx:
                d = abs(row[i] - old_val)
                removed.add(d)
        used_diffs.difference_update(removed)
        
        row[idx] = new_val
        
        added = set()
        for i in range(len(row)):
            if i != idx:
                d = abs(row[i] - new_val)
                added.add(d)
        used_diffs.update(added)

    best_scope = max(max(row) for row in rows)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        # Try to reduce the global maximum
        # Find rows with current max
        current_max = max(max(row) for row in rows)
        
        # Randomize row order to avoid bias
        row_indices = list(range(n_rows))
        random.shuffle(row_indices)
        
        for r in row_indices:
            if time.time() > deadline:
                break
            row = rows[r]
            # Try to move elements from largest to smallest
            # We only care about reducing the max, so focus on the last element first
            # But moving smaller elements might allow the last one to move.
            
            # Strategy: Try to find a smaller valid position for any element in this row
            # that reduces the row's max or the global max.
            
            # Specifically, try to minimize the last element
            # Scan down from row[-1]-1 to row[-2]+1
            # But we can also move other elements.
            
            # Let's try a generic move: for each idx in reversed(range(1, len(row))):
            #   try to move row[idx] to a smaller value
            
            for idx in reversed(range(1, len(row))):
                old_val = row[idx]
                # Try to find smallest valid new_val < old_val
                # Start from old_val - 1 down to row[idx-1] + 1
                found = False
                for new_val in range(old_val - 1, row[idx-1], -1):
                    if can_move(r, idx, new_val):
                        # Accept if it reduces the global max OR if it's the first improvement
                        # To be safe, always accept if it reduces the max.
                        # If it doesn't reduce the max, we might still accept to compact,
                        # but we need to be careful not to increase the max.
                        
                        # Calculate new max
                        # If idx is the last element, new_max_row = new_val
                        # Else, new_max_row = row[-1] (unchanged) unless we move last element
                        
                        do_move(r, idx, new_val)
                        improved = True
                        found = True
                        # Update best_scope
                        new_global_max = max(max(row) for row in rows)
                        if new_global_max < best_scope:
                            best_scope = new_global_max
                        break
                if found:
                    # Restart the row iteration? Or just continue?
                    # Since we changed the row, other indices might be affected.
                    # But we are scanning down.
                    # Let's just break out of idx loop and restart row loop?
                    # Or continue to next idx?
                    # Moving one element might enable moving another.
                    # Let's restart the inner loop for this row?
                    # No, just break and let the outer while loop handle it.
                    break
        
        if not improved:
            # Stuck. Try random moves?
            # Or restart with a different greedy seed?
            # Let's try a random perturbation: pick a random row, random element,
            # move it to a random valid position (not necessarily smaller) to escape local min.
            
            # But we must ensure we don't increase the max.
            # So, random move must result in max <= best_scope.
            
            # Pick random row
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            # Pick random element (not 0)
            idx = random.randint(1, len(row) - 1)
            
            # Find all valid positions for this element
            valid_positions = []
            # Range: row[idx-1] + 1 to row[idx+1] - 1 (or best_scope if last)
            upper_bound = best_scope
            if idx < len(row) - 1:
                upper_bound = min(upper_bound, row[idx+1] - 1)
            else:
                upper_bound = min(upper_bound, best_scope)
                
            lower_bound = row[idx-1] + 1
            
            # Sample some valid positions
            # Instead of checking all, just try a few random ones
            for _ in range(100):
                new_val = random.randint(lower_bound, upper_bound)
                if can_move(r, idx, new_val):
                    # Check if it increases the max
                    old_max = max(row)
                    # Simulate max
                    if idx == len(row) - 1:
                        new_max = new_val
                    else:
                        new_max = old_max
                    
                    if new_max <= best_scope:
                        do_move(r, idx, new_val)
                        improved = True
                        break
            
            if not improved:
                # If we are stuck for a while, break?
                # Or just continue hoping for a random hit?
                # Given the small size, we might just break to save time if no improvement.
                # But maybe we should try a few more iterations.
                pass

    return rows
