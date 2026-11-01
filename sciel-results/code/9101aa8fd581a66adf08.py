import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin under 120s kill

    def time_left():
        return deadline - time.time()

    # Pre-compute indices for difference calculation
    # For a row of length k+1, indices i < j
    diff_indices = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]

    def get_row_diffs(row):
        """Returns a set of differences for a given row."""
        diffs = set()
        for i, j in diff_indices:
            diffs.add(row[j] - row[i])
        return diffs

    def is_row_valid_internal(row):
        """Check if a row has strictly increasing elements and distinct internal differences."""
        # Strictly increasing check
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Distinct internal differences check
        diffs = set()
        for i, j in diff_indices:
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
        return True

    def greedy_row(used_diffs, max_val, jitter=0.0):
        """
        Greedily construct a row avoiding used_diffs.
        Returns None if impossible within max_val.
        """
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            # Look ahead window
            limit = min(max_val + 1, x + 100)
            for trial_x in range(x + 1, limit):
                new = {trial_x - a for a in row}
                if not (new & used_diffs) and not (new & row_diffs):
                    # Apply jitter: sometimes skip to explore
                    if jitter > 0 and random.random() < jitter and trial_x < 200:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                # If not found in lookahead, try strictly increasing until max_val
                # This is a fallback, usually fails if max_val is tight
                while x < max_val:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used_diffs) and not (new & row_diffs):
                        row.append(x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    return None
            x = row[-1]
        return row

    # Initial Solution: Greedy
    used_global = set()
    init_rows = []
    for _ in range(n_rows):
        # Greedy doesn't need jitter for init, just standard
        r = greedy_row(used_global, 10000, jitter=0.0)
        if r is None:
            # Fallback: simple greedy without max_val constraint
            row = [0]
            rd = set()
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used_global) and not (new & rd):
                        break
                    x += 1
                row.append(x)
                rd |= new
            r = row
        init_rows.append(r)
        used_global |= get_row_diffs(r)
    
    current_rows = [list(r) for r in init_rows]
    current_scope = max(max(r) for r in current_rows)
    
    # Track diffs for fast update
    current_row_diffs = [get_row_diffs(r) for r in current_rows]
    global_diffs = set()
    for d in current_row_diffs:
        global_diffs |= d

    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    def scope(rows):
        return max(max(r) for r in rows)

    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.9995 # Adjust based on total steps
    
    # Estimate steps: We want to use ~110 seconds.
    # Each step is fast (set operations on small sets).
    # Let's run until time runs out.
    
    step = 0
    while time_left() > 2:
        step += 1
        
        # Choose move type
        # 80% local element move, 20% row regeneration
        if random.random() < 0.8:
            # Local Element Move
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k) # Don't change index 0 (must be 0)
            
            old_val = current_rows[r_idx][p_idx]
            
            # Determine valid range for new value
            prev_val = current_rows[r_idx][p_idx - 1]
            next_val = current_rows[r_idx][p_idx + 1] if p_idx < k else current_scope + 50
            
            lower = prev_val + 1
            upper = min(next_val - 1, current_scope + 10) # Allow slight expansion
            
            if lower > upper:
                continue
                
            # Sample new value
            # Bias towards lower values to reduce scope
            if random.random() < 0.6:
                new_val = random.randint(lower, old_val)
            else:
                new_val = random.randint(lower, upper)
                
            if new_val == old_val:
                continue
            
            # Construct new row
            new_row = list(current_rows[r_idx])
            new_row[p_idx] = new_val
            
            # Check internal validity (strictly increasing and distinct diffs)
            # Since we only changed one element, we can check quickly
            # But easiest is to recompute diffs for this row
            new_row_diffs = get_row_diffs(new_row)
            
            # Check for internal duplicates? 
            # If len(new_row_diffs) != len(diff_indices), there are duplicates
            if len(new_row_diffs) != len(diff_indices):
                continue
            
            # Check against global diffs (excluding current row's old diffs)
            # We don't need to remove old diffs from global_diffs set if we just check intersection
            # But we must ensure new diffs don't conflict with OTHER rows' diffs
            # Other rows' diffs = global_diffs - current_row_diffs[r_idx]
            # Instead of set subtraction, we can just check:
            # If any diff in new_row_diffs is in global_diffs AND was not in current_row_diffs[r_idx], it's a conflict.
            
            # Optimization: 
            # Conflict if: d in new_row_diffs AND d in global_diffs AND d not in current_row_diffs[r_idx]
            # This is still O(N). 
            # Better: Maintain a count array or just use set difference?
            # Set difference is fast for small sets.
            other_diffs = global_diffs - current_row_diffs[r_idx]
            
            if new_row_diffs & other_diffs:
                continue
            
            # Valid move!
            # Calculate new scope
            new_scope = max(max(current_rows[i] if i != r_idx else new_row) for i in range(n_rows))
            
            # Metropolis Acceptance
            delta = new_scope - current_scope
            if delta < 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    accept = random.random() < 0.5 # Accept neutral moves sometimes to shuffle
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                # Update state
                current_rows[r_idx] = new_row
                current_row_diffs[r_idx] = new_row_diffs
                # Update global_diffs
                global_diffs.difference_update(current_row_diffs[r_idx]) # This is wrong if we just assigned. 
                # Correct update:
                # Remove old, add new
                # Since we already checked validity, we can just do:
                # global_diffs = (global_diffs - old_diffs) | new_diffs
                # But we modified current_row_diffs already.
                # Let's revert to safe update:
                # 1. Remove old diffs from global
                # 2. Add new diffs to global
                # We need the OLD diffs. We have them in the variable 'current_row_diffs' before assignment? 
                # No, we assigned new_row_diffs to current_row_diffs[r_idx] in the check phase? No, we didn't.
                # Let's re-structure the update block.
                
                # Re-do update logic safely:
                old_diffs = current_row_diffs[r_idx]
                
                # Update global set
                global_diffs.difference_update(old_diffs)
                global_diffs.update(new_row_diffs)
                
                # Update row and diffs
                current_rows[r_idx] = new_row
                current_row_diffs[r_idx] = new_row_diffs
                current_scope = new_scope
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                    
        else:
            # Row Regeneration Move
            r_idx = random.randint(0, n_rows - 1)
            
            # Determine max_val for new row
            # We want to try to keep scope low, but allow some expansion
            # Use current scope as a hard cap for the greedy, but allow it to fail if too tight?
            # If it fails, we skip the move.
            max_val = current_scope
            
            # Randomized greedy with jitter
            # Jitter helps explore different structures
            jitter = random.uniform(0.0, 0.3)
            
            # Get diffs of other rows
            other_diffs = global_diffs - current_row_diffs[r_idx]
            
            new_row = greedy_row(other_diffs, max_val, jitter=jitter)
            
            if new_row is None:
                continue
            
            new_row_diffs = get_row_diffs(new_row)
            
            # Check internal validity (greedy_row should ensure this, but double check)
            if len(new_row_diffs) != len(diff_indices):
                continue
            
            # Check against other diffs (already done in greedy_row via 'used_diffs' parameter? 
            # Yes, greedy_row checks 'new & used_diffs'. So new_row_diffs should be disjoint from other_diffs.
            # But let's verify just in case.
            if new_row_diffs & other_diffs:
                continue
            
            # Calculate new scope
            new_scope = max(max(current_rows[i] if i != r_idx else new_row) for i in range(n_rows))
            
            # Metropolis Acceptance
            delta = new_scope - current_scope
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                # Update state
                old_diffs = current_row_diffs[r_idx]
                global_diffs.difference_update(old_diffs)
                global_diffs.update(new_row_diffs)
                
                current_rows[r_idx] = new_row
                current_row_diffs[r_idx] = new_row_diffs
                current_scope = new_scope
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]

        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If T is very low, we might be stuck. 
        # If no improvement for a long time, restart T?
        # Simple restart: if current_scope == best_scope for a long time, maybe kick.
        # For now, the 20% row regeneration acts as a kick.

    # Final Polish: Greedy refinement of the best row
    # Try to reduce scope of best_rows by re-greedy-ing each row with current scope as cap
    polished_rows = [list(r) for r in best_rows]
    polished_diffs = [get_row_diffs(r) for r in polished_rows]
    p_global = set()
    for d in polished_diffs:
        p_global |= d
        
    improved = True
    while improved and time_left() > 1:
        improved = False
        for i in range(n_rows):
            # Try to rebuild row i with a tighter cap?
            # Current scope is max(polished_rows)
            # We can try to rebuild with cap = max(polished_rows[i]) - 1 ?
            # Or just standard greedy with jitter to see if we get a smaller max.
            other_diffs = p_global - polished_diffs[i]
            # Try a few times with different jitter
            for _ in range(5):
                jitter = random.uniform(0.0, 0.5)
                # Cap at current max of the row to see if we can shrink it
                cap = max(polished_rows[i])
                new_row = greedy_row(other_diffs, cap, jitter=jitter)
                if new_row is not None:
                    new_diffs = get_row_diffs(new_row)
                    if new_diffs & other_diffs:
                        continue
                    # Check if this reduces the global scope
                    # The new row's max is max(new_row)
                    # If max(new_row) < max(polished_rows[i]), it might reduce global scope
                    # But even if it doesn't, it might allow other rows to shrink?
                    # For simplicity, only accept if it reduces the max of this row
                    if max(new_row) < max(polished_rows[i]):
                        # Update
                        p_global.difference_update(polished_diffs[i])
                        p_global.update(new_diffs)
                        polished_rows[i] = new_row
                        polished_diffs[i] = new_diffs
                        improved = True
                        break
                    elif max(new_row) == max(polished_rows[i]) and random.random() < 0.1:
                        # Accept neutral changes to explore
                        p_global.difference_update(polished_diffs[i])
                        p_global.update(new_diffs)
                        polished_rows[i] = new_row
                        polished_diffs[i] = new_diffs
                        break

    if scope(polished_rows) < best_scope:
        best_rows = polished_rows
        best_scope = scope(best_rows)

    # Final validity check
    # Re-verify best_rows
    seen = set()
    valid = True
    for r in best_rows:
        if r[0] != 0:
            valid = False
            break
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                valid = False
                break
        if not valid: break
        rd = get_row_diffs(r)
        if len(rd) != len(diff_indices):
            valid = False
            break
        if rd & seen:
            valid = False
            break
        seen |= rd

    if not valid:
        # Fallback to initial greedy
        return init_rows

    return best_rows
