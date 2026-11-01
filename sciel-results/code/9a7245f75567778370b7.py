import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_solution = None
    best_scope = float('inf')
    
    # Counting bound for (7, 5) is 105. 
    # If we find 105, it's optimal.
    lower_bound = 105 
    
    # We will try to find a solution with scope < best_scope.
    # We use randomized restarts.
    
    def backtrack(rows, used_diffs, row_idx):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False, None
            
        if row_idx == n_rows:
            # All rows constructed
            # Calculate scope
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_solution = [list(r) for r in rows]
                # If we hit the lower bound, we can stop
                if best_scope <= lower_bound:
                    return True, None
            return False, None
            
        # Heuristic: Try to fill the current row.
        # To increase diversity, we can randomize the choice of next mark slightly,
        # but primarily we want small marks.
        
        row = [0]
        
        # We need k more marks.
        # We use a recursive helper to fill the row.
        
        def fill_row(current_row, current_used):
            # current_row is a list, current_used is a set of diffs for THIS row + global used?
            # No, we need to check against global used_diffs + diffs within current_row.
            
            if len(current_row) == k + 1:
                # Check if this row is valid with global used_diffs
                # The diffs within the row are already checked during construction.
                # We just need to ensure no conflict with global used_diffs.
                # Since we checked at each step, it's valid.
                return True
                
            last_mark = current_row[-1]
            
            # Candidate range: last_mark + 1 to best_scope (or some upper limit)
            # We don't know best_scope yet if it's inf, but we can use a heuristic limit.
            # Let's use best_scope if set, else a generous limit like 200.
            max_val = best_scope if best_scope != float('inf') else 200
            
            candidates = list(range(last_mark + 1, max_val + 1))
            
            # Randomize candidates to escape local minima?
            # Shuffling might be good.
            random.shuffle(candidates)
            
            # However, we prefer small values. 
            # A compromise: sort by value, but with a random tie-breaker or random selection from a prefix?
            # Let's just shuffle. The backtracking will handle the rest.
            # Actually, for minimization, greedy-small is usually better.
            # Let's try sorting by value, but adding a small random noise?
            # Or just shuffle.
            
            for x in candidates:
                # Check if x creates a conflict
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in current_used: # global used
                        conflict = True
                        break
                    if d in new_diffs: # internal duplicate? (shouldn't happen if row is increasing and we check against current_row)
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if not conflict:
                    # Recurse
                    current_row.append(x)
                    # Update current_used? 
                    # We pass a new set to avoid mutation issues, or mutate and backtrack.
                    # Mutating is faster.
                    new_global_used = current_used.union(new_diffs)
                    
                    # Also need to check internal diffs of the row?
                    # The loop above checks against `current_used` (global).
                    # It does NOT check against diffs already in `current_row` that are NOT in `current_used`.
                    # Wait, `current_used` is the global set.
                    # The diffs within `current_row` are NOT in `current_used` yet.
                    # So we must check against `new_diffs` (diffs with previous marks in this row) AND `current_used`.
                    # The loop `if d in current_used` checks global.
                    # The loop `if d in new_diffs` checks internal.
                    # This is correct.
                    
                    if fill_row(current_row, new_global_used):
                        return True
                    
                    current_row.pop()
                
            return False
        
        # Start filling row
        if fill_row(row, used_diffs):
            # Row is complete
            rows.append(row)
            # Update global used_diffs
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            # Recurse to next row
            if backtrack(rows, used_diffs.union(row_diffs), row_idx + 1):
                return True, None
            
            rows.pop()
        
        return False, None

    # Main loop: Randomized restarts
    while time.time() < deadline:
        # Shuffle row order? The rows are identical in requirement.
        # But the sequence of used_diffs matters.
        # We can just start fresh each time.
        
        # To add diversity, we can randomize the initial "bias" or just rely on random.shuffle in candidates.
        
        rows = []
        used_diffs = set()
        
        # We call backtrack. 
        # Note: backtrack modifies best_solution if it finds a better one.
        # We want it to find ANY valid solution, then we compare scope.
        # The backtrack function above tries to find a solution with scope < best_scope?
        # No, it just finds a solution. The `max_val` limit uses `best_scope`.
        # So if `best_scope` is 112, it will only look for solutions with max < 112.
        # This is a good pruning strategy.
        
        # However, if `best_scope` is inf, it looks up to 200.
        
        # We need to reset the "found better" logic?
        # The nonlocal variables persist.
        
        # Let's run one backtracking attempt.
        # If it finds a solution, it updates best_solution.
        # If it fails, we try again.
        
        # To speed up, we can limit the depth or width?
        # The backtracking is exponential.
        # For 7 rows, it might be fast if pruning is good.
        
        backtrack(rows, used_diffs, 0)
        
        # If we found the lower bound, break
        if best_scope <= lower_bound:
            break
        
        # Small sleep to avoid burning CPU if backtracking is too fast and failing?
        # No, backtracking takes time.
        
    if best_solution is None:
        # Fallback to greedy if no solution found (should not happen)
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
        
    return best_solution
