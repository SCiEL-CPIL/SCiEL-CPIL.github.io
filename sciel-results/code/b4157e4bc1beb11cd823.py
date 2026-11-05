import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Parameters specific to the task (4 rows, k=7 -> 8 elements)
    # But the function should be general.
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_used_diffs(rows, max_diff):
        used = np.zeros(max_diff + 1, dtype=bool)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True
        return used

    def greedy_construct(scope_limit=200):
        """
        Greedy construction with backtracking for the last few elements if stuck.
        """
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            # Greedily pick next elements
            for pos in range(1, k + 1):
                found = False
                last_val = row[-1]
                # Try to find a value that doesn't conflict
                # We can start from last_val + 1 up to some limit
                # A good heuristic is to try small gaps first
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    # If greedy fails, we could backtrack, but for simplicity, 
                    # we'll return None and let the caller try a different scope or random restart
                    return None
            rows.append(row)
            
        return rows

    def local_search(initial_rows, time_limit):
        """
        Local search with single element moves and suffix shifts.
        """
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        max_diff_bound = current_scope + 100
        used_diffs = compute_used_diffs(rows, max_diff_bound)
        
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        t_end = time.time() + time_limit
        no_improve_count = 0
        max_no_improve = 5000  # Restart local search if stuck
        
        while time.time() < t_end:
            if no_improve_count > max_no_improve:
                # Perturb the current solution to escape local minimum
                # Randomly change a few elements
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    old_val = rows[r_idx][pos]
                    # Try to find a new valid value for this position
                    min_val = rows[r_idx][pos - 1] + 1
                    max_val = rows[r_idx][pos + 1] - 1 if pos < k else best_scope + 20
                    if min_val <= max_val:
                        # Remove old diffs
                        old_diffs = []
                        for i in range(k + 1):
                            if i != pos:
                                if i < pos:
                                    d = old_val - rows[r_idx][i]
                                else:
                                    d = rows[r_idx][i] - old_val
                                old_diffs.append(d)
                                if d <= max_diff_bound:
                                    used_diffs[d] = False
                        
                        # Try new values
                        found = False
                        for x in random.sample(range(min_val, max_val + 1), min(10, max_val - min_val + 1)):
                            if x == old_val: continue
                            new_diffs = []
                            valid = True
                            for i in range(k + 1):
                                if i != pos:
                                    if i < pos:
                                        d = x - rows[r_idx][i]
                                    else:
                                        d = rows[r_idx][i] - x
                                    if d > max_diff_bound or used_diffs[d]:
                                        valid = False
                                        break
                                    new_diffs.append(d)
                            if valid:
                                rows[r_idx][pos] = x
                                for d in new_diffs:
                                    used_diffs[d] = True
                                current_scope = max(current_scope, get_scope(rows))
                                found = True
                                break
                        if not found:
                            # Restore old
                            rows[r_idx][pos] = old_val
                            for d in old_diffs:
                                if d <= max_diff_bound:
                                    used_diffs[d] = True
                
                no_improve_count = 0
                continue

            # Choose a move type: 0 = single element, 1 = suffix shift
            move_type = random.randint(0, 1)
            
            if move_type == 0:
                # Single element move
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                
                old_val = rows[row_idx][pos]
                min_val = rows[row_idx][pos - 1] + 1
                max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 20
                
                if min_val > max_val:
                    no_improve_count += 1
                    continue

                # Try a few random candidates
                candidates = random.sample(range(min_val, max_val + 1), min(5, max_val - min_val + 1))
                improved = False
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    
                    # Collect old diffs
                    old_diffs = []
                    for i in range(k + 1):
                        if i != pos:
                            if i < pos:
                                d = old_val - rows[row_idx][i]
                            else:
                                d = rows[row_idx][i] - old_val
                            old_diffs.append(d)
                            if d <= max_diff_bound:
                                used_diffs[d] = False
                    
                    # Check new diffs
                    new_diffs = []
                    valid = True
                    for i in range(k + 1):
                        if i != pos:
                            if i < pos:
                                d = new_val - rows[row_idx][i]
                            else:
                                d = rows[row_idx][i] - new_val
                            if d > max_diff_bound or used_diffs[d]:
                                valid = False
                                break
                            new_diffs.append(d)
                    
                    if valid:
                        # Calculate new scope
                        # Only need to check if this row's max changed
                        # And if the global max changed
                        old_row_max = max(rows[row_idx])
                        new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                        
                        # Global scope is max of all rows
                        # If this row was the unique max, scope might decrease
                        # If this row is not the max, scope stays same unless new_val > current_scope
                        if new_row_max < old_row_max:
                            # Recalculate global scope
                            other_max = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
                            new_scope = max(other_max, new_row_max)
                        else:
                            new_scope = current_scope
                            if new_row_max > current_scope:
                                new_scope = new_row_max
                        
                        if new_scope < current_scope:
                            # Accept
                            rows[row_idx][pos] = new_val
                            for d in new_diffs:
                                used_diffs[d] = True
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                            no_improve_count = 0
                            improved = True
                            break
                        else:
                            # Reject: restore
                            for d in old_diffs:
                                if d <= max_diff_bound:
                                    used_diffs[d] = True
                    else:
                        # Restore
                        for d in old_diffs:
                            if d <= max_diff_bound:
                                used_diffs[d] = True
                
                if not improved:
                    no_improve_count += 1
            else:
                # Suffix shift
                row_idx = random.randint(0, n_rows - 1)
                # Choose a split point j (1 <= j <= k-1)
                # Shift elements j..k by delta
                if k < 2:
                    no_improve_count += 1
                    continue
                j = random.randint(1, k - 1)
                
                # Determine valid delta range
                # Elements 0..j-1 are fixed. Elements j..k are shifted.
                # Constraint: rows[row_idx][j-1] < rows[row_idx][j] + delta
                # Constraint: rows[row_idx][k] + delta <= best_scope + 20 (or current_scope + 20)
                
                min_delta = rows[row_idx][j-1] - rows[row_idx][j] + 1
                max_delta = (best_scope + 20) - rows[row_idx][k]
                
                if min_delta > max_delta:
                    no_improve_count += 1
                    continue
                
                # Try a few random deltas
                candidates = random.sample(range(min_delta, max_delta + 1), min(5, max_delta - min_delta + 1))
                improved = False
                for delta in candidates:
                    if delta == 0:
                        continue
                    
                    # Identify affected differences
                    # Differences involving at least one index >= j and at least one index < j
                    # These are the ones that change.
                    # Differences within 0..j-1 are unchanged.
                    # Differences within j..k are unchanged.
                    
                    # We need to remove the old "crossing" differences and add the new ones.
                    old_cross_diffs = []
                    for i in range(j):
                        for l in range(j, k + 1):
                            d = rows[row_idx][l] - rows[row_idx][i]
                            old_cross_diffs.append(d)
                            if d <= max_diff_bound:
                                used_diffs[d] = False
                    
                    # Check if new crossing differences are valid
                    new_cross_diffs = []
                    valid = True
                    for i in range(j):
                        for l in range(j, k + 1):
                            d = (rows[row_idx][l] + delta) - rows[row_idx][i]
                            if d > max_diff_bound or used_diffs[d]:
                                valid = False
                                break
                            new_cross_diffs.append(d)
                        if not valid:
                            break
                    
                    if valid:
                        # Check scope
                        new_last = rows[row_idx][k] + delta
                        # Since the row is increasing, the max of the row is the last element
                        # But wait, if delta is negative, the last element decreases.
                        # If delta is positive, the last element increases.
                        # The max of the row is always the last element because it's increasing.
                        # So new_row_max = new_last.
                        
                        old_row_max = rows[row_idx][k]
                        new_row_max = new_last
                        
                        if new_row_max < old_row_max:
                            other_max = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
                            new_scope = max(other_max, new_row_max)
                        else:
                            new_scope = current_scope
                            if new_row_max > current_scope:
                                new_scope = new_row_max
                        
                        if new_scope < current_scope:
                            # Apply shift
                            for l in range(j, k + 1):
                                rows[row_idx][l] += delta
                            for d in new_cross_diffs:
                                used_diffs[d] = True
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                            no_improve_count = 0
                            improved = True
                            break
                        else:
                            # Reject: restore old crossing diffs
                            for d in old_cross_diffs:
                                if d <= max_diff_bound:
                                    used_diffs[d] = True
                    else:
                        # Reject: restore old crossing diffs
                        for d in old_cross_diffs:
                            if d <= max_diff_bound:
                                used_diffs[d] = True
                
                if not improved:
                    no_improve_count += 1
        
        return best_rows, best_scope

    # Step 1: Generate initial solution
    # Try greedy with increasing scope limits
    initial_rows = None
    for scope_limit in [120, 130, 140, 150, 200, 300]:
        initial_rows = greedy_construct(scope_limit)
        if initial_rows is not None:
            break
    
    if initial_rows is None:
        # Fallback: Very simple construction
        used_diffs = set()
        initial_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for y in row:
                    if x - y in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for y in row[:-1]:
                        used_diffs.add(x - y)
            initial_rows.append(row)
        return initial_rows

    current_scope = get_scope(initial_rows)
    
    # Step 2: Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        best_rows, best_scope = local_search(initial_rows, remaining_time - 2)
        if is_valid(best_rows):
            return best_rows
        else:
            return initial_rows
    
    return initial_rows

if __name__ == "__main__":
    # Test with the specific task parameters
    rows = construct(4, 7)
    scope = max(max(r) for r in rows)
    print(f"Scope: {scope}")
    print(rows)
