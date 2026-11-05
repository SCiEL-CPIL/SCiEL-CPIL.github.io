import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # The known best for (4,7) is 134. Let's start with a reasonable bound.
    # If the greedy construction exceeds this, we increase the bound.
    max_scope_limit = 2000

    def greedy_construct(scope_limit):
        """
        Greedy construction: build rows one by one, each row greedily.
        Uses a set for speed during construction (simplest), then converts.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                # Try small increments first
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
                    return None
            rows.append(row)
        return rows

    def validate(rows):
        """Check if all differences are distinct across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated Annealing to minimize the scope (largest entry).
        Moves:
        1. Shrink: Try to reduce the last element of a row.
        2. Move: Try to move an internal element to a smaller valid position.
        3. Random perturbation to escape local minima.
        """
        rows = [r[:] for r in initial_rows]
        
        # Determine max scope for array
        current_max = max(max(r) for r in rows)
        arr_size = max(current_max + 200, 500)
        
        # Initialize difference array
        diff_arr = np.zeros(arr_size, dtype=bool)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d < arr_size:
                        diff_arr[d] = True
        
        def compute_state():
            """Compute the current scope and validity."""
            scope = max(max(r) for r in rows)
            # Check validity using diff_arr (should be valid if we maintain it correctly)
            # But for safety in SA, we might have invalid states.
            # For now, assume we only accept valid moves or handle invalidity via penalty.
            # Let's just return scope.
            return scope

        def try_move(row_idx, pos, new_val):
            """
            Try to change rows[row_idx][pos] to new_val.
            Returns (success, delta_scope) or None if invalid.
            """
            row = rows[row_idx]
            old_val = row[pos]
            if old_val == new_val:
                return None
            
            # 1. Collect old diffs and clear them
            old_diffs = []
            for i in range(len(row)):
                if i != pos:
                    if i < pos:
                        d = old_val - row[i]
                    else:
                        d = row[i] - old_val
                    old_diffs.append(d)
            
            # Check bounds
            if new_val < 0 or new_val >= arr_size:
                return None
            
            # Clear old diffs
            for d in old_diffs:
                if d < arr_size:
                    diff_arr[d] = False
            
            # 2. Check new diffs
            new_diffs = []
            for i in range(len(row)):
                if i != pos:
                    if i < pos:
                        # Must maintain increasing order: row[i] < new_val
                        if row[i] >= new_val:
                            # Restore and fail
                            for od in old_diffs:
                                if od < arr_size:
                                    diff_arr[od] = True
                            return None
                        d = new_val - row[i]
                    else:
                        # Must maintain increasing order: new_val < row[i]
                        if row[i] <= new_val:
                            # Restore and fail
                            for od in old_diffs:
                                if od < arr_size:
                                    diff_arr[od] = True
                            return None
                        d = row[i] - new_val
                    
                    if d >= arr_size:
                        for od in old_diffs:
                            if od < arr_size:
                                diff_arr[od] = True
                        return None
                    
                    if diff_arr[d]:
                        # Conflict!
                        for od in old_diffs:
                            if od < arr_size:
                                diff_arr[od] = True
                        return None
                    new_diffs.append(d)
            
            # 3. Commit new diffs
            for d in new_diffs:
                diff_arr[d] = True
            
            # Update row
            row[pos] = new_val
            
            # Calculate delta scope
            old_scope = max(max(r) for r in rows) # This is expensive, optimize later
            # Actually, we can track scope more efficiently, but for now this is okay for small n.
            # Better: just return the new scope.
            new_scope = max(max(r) for r in rows)
            
            return (True, new_scope)

        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in rows)
        current_scope = best_scope
        
        t_start = time.time()
        t_end = t_start + time_limit
        
        # Initial temperature
        T = 10.0
        cooling_rate = 0.99
        
        while time.time() < t_end:
            # Pick a random row and position (except 0, which is fixed)
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(rows[r_idx]) - 1)
            
            row = rows[r_idx]
            old_val = row[pos]
            
            # Determine range for new value
            min_v = row[pos - 1] + 1
            max_v = row[pos + 1] - 1 if pos < len(row) - 1 else best_scope + 100
            
            if min_v > max_v:
                continue
            
            # Choose a new value
            # Bias towards smaller values to reduce scope
            # Or random within range
            if random.random() < 0.5:
                # Try to shrink
                new_val = random.randint(min_v, min(old_val, max_v))
            else:
                # Random move
                new_val = random.randint(min_v, max_v)
            
            if new_val == old_val:
                continue
            
            # Try the move
            result = try_move(r_idx, pos, new_val)
            
            if result is None:
                # Move invalid, revert (try_move already reverts on failure)
                continue
            
            success, new_scope = result
            
            if success:
                delta_scope = new_scope - current_scope
                # Accept if better, or with probability exp(-delta/T)
                if delta_scope < 0 or random.random() < math.exp(-delta_scope / T):
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    # Revert move
                    # We need to revert the move we just made
                    # try_move doesn't support revert directly, so we do it manually
                    # Actually, let's just call try_move with old_val to revert?
                    # No, that's messy. Let's structure try_move to return a revert function or handle it.
                    # For simplicity in this SA, let's just re-evaluate and revert if rejected.
                    # Revert:
                    old_diffs = []
                    for i in range(len(row)):
                        if i != pos:
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            old_diffs.append(d)
                            diff_arr[d] = False
                    
                    for d in old_diffs:
                        if d < arr_size:
                            diff_arr[d] = True # Wait, these are the NEW diffs we just added. We need to remove them.
                            # Correction: We added new_diffs to diff_arr. We need to clear them.
                            pass # The loop above clears them.
                    
                    # Restore old diffs
                    for i in range(len(row)):
                        if i != pos:
                            if i < pos:
                                d = old_val - row[i]
                            else:
                                d = row[i] - old_val
                            if d < arr_size:
                                diff_arr[d] = True
                    
                    row[pos] = old_val
                    current_scope = max(max(r) for r in rows) # Recalculate
            
            # Cool down
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
        
        return best_rows, best_scope

    # Step 1: Greedy construction
    best_rows = None
    best_scope = float('inf')

    # Try to construct with a tight scope first
    # The counting bound is 112. Known best is 134.
    # Let's try to construct greedily and see what scope we get.
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
            
        rows = greedy_construct(scope_limit=500)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
    if best_rows is None:
        # Fallback to larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
        else:
            # Emergency fallback: just return something valid, even if large
            used_diffs = set()
            rows = []
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
                rows.append(row)
            return rows

    # Step 2: Local Search / Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = simulated_annealing(best_rows, remaining_time - 5)

    return best_rows
