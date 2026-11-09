"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Build up to k+1 elements in each row
    for step in range(1, k + 1):
        # Shuffle row order for diversity
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _optimize(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Optimized local search using difference counts.
    Returns the best rows found.
    """
    random.seed(seed)
    start_time = time.time()
    
    # Pre-compute max possible difference to size the count array
    # Heuristic: max entry is likely < 300 for these small problems
    # We can dynamically resize if needed, but let's start with a safe bound
    # Actually, let's just use a dictionary or a large enough list.
    # Given k=5, n_rows=7, total diffs = 7 * 15 = 105.
    # Max diff is bounded by the largest element.
    # Let's use a list of size 1000 for speed. If we exceed it, we extend.
    
    current_rows = [list(r) for r in rows]
    
    # Helper to compute current scope
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Initialize difference counts
    # We need to track which differences are used.
    # Since we only care about uniqueness, count > 1 is invalid.
    # We maintain a count array.
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 10
    diff_counts = [0] * (diff_size + 1)
    
    # Populate initial counts
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    # Extend if necessary (shouldn't happen often)
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    # Check if initial state is valid (it should be from greedy)
    # But just in case, if any count > 1, we have a problem.
    # Greedy ensures validity.
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    # Local Search: Hill Climbing
    # We try to reduce the scope by moving elements to smaller positions.
    # A move is valid if:
    # 1. The row remains strictly increasing.
    # 2. The new differences introduced do not conflict with existing differences
    #    (i.e., count of new diff was 0, or count was 1 and it was from this same pair in this row? 
    #    No, differences are global. If a diff exists in ANY other row, count > 0.
    #    If a diff exists in THIS row from a DIFFERENT pair, count > 0.
    #    So, for a move to be valid, all NEW differences must have count 0 BEFORE the move?
    #    Wait. The old differences from this position are removed.
    #    So, for each new diff d:
    #    - If d was one of the old diffs for this position, its count decreases by 1 (from old) and increases by 1 (new) -> Net 0 change for that specific d?
    #      No, the set of old diffs is D_old, new diffs is D_new.
    #      We remove D_old from counts, add D_new to counts.
    #      Conflict if any d in D_new has count > 1 AFTER removal of D_old and addition of D_new?
    #      Actually, simpler:
    #      1. Temporarily remove D_old from counts.
    #      2. Check if any d in D_new has count > 0. If yes, conflict (because that d is used elsewhere or by another pair in this row).
    #         Note: If d is in D_old, its count was decreased. If it's the ONLY source, count becomes 0.
    #         If d is in D_new and d was in D_old, and count was 1 (only this pair), then after removal count=0, so adding it back is fine.
    #         If d is in D_new and d was NOT in D_old, count must be 0.
    #      3. If valid, add D_new to counts.
    #      4. If invalid, restore D_old.
    
    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        
        # Check increasing constraint
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Calculate old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old diffs
        for d in old_diffs:
            diff_counts[d] -= 1
            
        # Check new diffs
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                 # This shouldn't happen if we sized correctly, but handle it
                 pass
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old diffs
            for d in old_diffs:
                diff_counts[d] += 1
            return False
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        return True

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate rows and positions in random order for diversity
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            # Try to reduce elements from right to left? Or random?
            # Let's try to reduce the largest elements first as they contribute most to scope.
            # But we must maintain order.
            # Let's just iterate all positions.
            for pos in range(len(row) - 1, 0, -1): # Right to left
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                # Try to reduce to smallest valid value
                # Lower bound: row[pos-1] + 1 (if pos > 0)
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                # Upper bound: old_val - 1
                # We want the smallest new_val >= lower_bound such that move is valid.
                # Actually, we want to minimize the scope, so we want the smallest possible value.
                # However, changing to a smaller value might increase other differences?
                # No, differences are absolute values.
                # If we move x to y < x, differences with elements < x change, with elements > x change.
                
                # Optimization: Just try values from lower_bound up to old_val - 1?
                # Or from old_val - 1 down?
                # To minimize scope, we generally want to pull elements down.
                # Let's try from lower_bound upwards to find the *smallest* valid position.
                # This might be slow if the gap is large.
                # Better: Try from old_val - 1 downwards? No, that finds the *largest* valid position < old_val.
                # We want to minimize the max entry. So for each element, we want it as small as possible.
                
                found = False
                # Try values from lower_bound to old_val - 1
                # If the range is huge, this is slow.
                # But old_val is bounded by current scope (~100-200).
                for new_val in range(lower_bound, old_val):
                    if try_move(r, pos, new_val):
                        improved = True
                        found = True
                        break # Found smallest valid, stop trying larger (worse) values for this pos?
                        # Wait. If we find a valid new_val, we should take it.
                        # Is it always better to take the SMALLEST valid new_val?
                        # Yes, for that element. But it might block others?
                        # Hill climbing assumes local improvement.
                        # Taking the smallest valid value reduces the element's value, which can only help or be neutral for future reductions of this element.
                        # It might increase differences with other elements, potentially causing conflicts later.
                        # But standard greedy reduction works well.
                        break
                if not found:
                    # No smaller value works. Try to see if we can move it UP? No, we want to minimize scope.
                    # If we can't reduce it, leave it.
                    pass
                    
    # Update best
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple seeds for greedy initialization
    # Limit number of restarts to ensure time for optimization
    max_restarts = 10
    for seed in range(max_restarts):
        if time.time() > deadline - 30.0: # Leave 30s for optimization
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback
        rows = []
        used = set()
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

    # Optimize the best found solution
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        best_rows = _optimize(best_rows, n_rows, k, time_limit=time_remaining, seed=42)
        
    return best_rows
# EVOLVE-BLOCK-END
