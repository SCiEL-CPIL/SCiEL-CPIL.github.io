import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Total number of differences needed
    # Each row has k+1 entries -> (k+1)*k/2 differences
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # 1. Greedy Initial Solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if x - a in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Helper to remove row diffs from used
    def remove_row_diffs(row, used):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i):
                used.discard(ri - row[j])
                
    # Helper to add row diffs to used
    def add_row_diffs(row, used):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i):
                used.add(ri - row[j])
                
    # Helper to check if a row is valid given the used set (excluding its own diffs)
    def is_row_valid(row, used):
        row_diffs = []
        for i in range(len(row)):
            ri = row[i]
            for j in range(i):
                d = ri - row[j]
                if d in used:
                    return False
                row_diffs.append(d)
        return True

    # 2. Simulated Annealing Optimization
    # We will perform random moves:
    # A. Rebuild a single row from scratch (removing its diffs, finding a new valid row)
    # B. Perturb a single row (change one element, check validity, accept/reject)
    
    # To make "Rebuild" fast, we use a heuristic greedy that tries to minimize the max element
    # but allows some randomness to escape local minima.
    
    def try_rebuild_row(idx, used, max_val_limit):
        """
        Attempts to find a new row for index idx that is valid with current 'used'.
        Returns the new row if found, else None.
        It tries to keep the max value low.
        """
        # We use a randomized greedy approach
        # Sort candidates by "cost" (distance to existing elements?)
        
        # Let's try a few random restarts of a greedy builder
        best_local_row = None
        best_local_max = float('inf')
        
        # Limit attempts to keep it fast
        for _ in range(5):
            row = [0]
            current_max_val = 0
            valid = True
            
            # Randomly decide if we try to be "tight" (small increments) or "loose"
            # Tight: try smallest possible x
            # Loose: try x with some randomness
            
            for step in range(k):
                # Determine range of x to try
                # Min x is row[-1] + 1
                # Max x is bounded by max_val_limit (or best_local_max - 1)
                min_x = row[-1] + 1
                max_x = max_val_limit if max_val_limit < best_local_max else best_local_max - 1
                if max_x < min_x:
                    valid = False
                    break
                
                # Try candidates in increasing order, but with a small random skip probability
                # to avoid getting stuck in the exact same local minimum
                found = False
                # We iterate up to a certain limit to avoid infinite loops if max_x is huge
                # But max_x is bounded by best_local_max which is finite.
                
                # Optimization: Instead of checking every x from min_x to max_x,
                # we can check a subset first, or just go sequentially.
                # Since k is small (6), the number of diffs per row is 21.
                # The search space is small enough that sequential check is okay if we prune well.
                
                for x in range(min_x, max_x + 1):
                    # Check validity
                    ok = True
                    for a in row:
                        if x - a in used:
                            ok = False
                            break
                    if ok:
                        # Tentatively add
                        row.append(x)
                        # Add diffs to a temporary set? No, we need to check against 'used' only.
                        # The internal diffs of the row don't conflict with 'used' yet.
                        # But we must ensure internal diffs are unique? 
                        # The problem statement says "all differences between two entries of the same row are distinct over all rows".
                        # This implies:
                        # 1. Diffs within a row are distinct from each other?
                        #    "distinct over all rows" usually means the global set of all diffs (from all rows) has no duplicates.
                        #    If a row has duplicate diffs (e.g., 0, 1, 2 -> diffs 1, 2, 1), then the global set has duplicates.
                        #    So yes, internal diffs must be distinct.
                        #    My current check `if x - a in used` only checks against 'used' (other rows).
                        #    I MUST also check against diffs already in `row`.
                        
                        # Let's re-verify the logic.
                        # `used` contains diffs from all OTHER rows.
                        # `row` contains the current partial row.
                        # We need `x - a` to not be in `used` AND not in any `b - c` for `b, c` in `row` (where `b > c`).
                        
                        # Let's collect existing diffs in `row`
                        row_diffs_set = set()
                        for i in range(len(row)): # row currently has len(row) elements (before append)
                            ri = row[i]
                            for j in range(i):
                                row_diffs_set.add(ri - row[j])
                        
                        # Check new diffs
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used or d in row_diffs_set:
                                ok = False
                                break
                            new_diffs.append(d)
                        
                        if ok:
                            # Accept x
                            row.append(x)
                            found = True
                            break
                        else:
                            # Reject x, continue to next x
                            pass
                
                if not found:
                    valid = False
                    break
            
            if valid:
                curr_max = row[-1]
                if curr_max < best_local_max:
                    best_local_max = curr_max
                    best_local_row = row[:]
                    
                # If we found a very good one, we can stop
                if best_local_max == min_x_initial: # theoretical min
                    break
                    
        return best_local_row

    # Note: The `try_rebuild_row` above is a bit slow because it recalculates `row_diffs_set` in the inner loop.
    # Let's optimize it.
    
    def try_rebuild_row_optimized(idx, used, max_val_limit):
        best_local_row = None
        best_local_max = float('inf')
        
        # Number of restarts for randomized greedy
        restarts = 10
        for _ in range(restarts):
            row = [0]
            row_diffs = set()
            valid = True
            
            for step in range(k):
                min_x = row[-1] + 1
                max_x = max_val_limit if max_val_limit < best_local_max else best_local_max - 1
                if max_x < min_x:
                    valid = False
                    break
                
                found = False
                # Randomization: sometimes start from a higher x
                start_x = min_x
                if random.random() < 0.2:
                    start_x = min_x + random.randint(0, 5)
                    if start_x > max_x:
                        start_x = min_x
                
                for x in range(start_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used or d in row_diffs:
                            ok = False
                            break
                    if ok:
                        # Add x
                        for a in row:
                            row_diffs.add(x - a)
                        row.append(x)
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                curr_max = row[-1]
                if curr_max < best_local_max:
                    best_local_max = curr_max
                    best_local_row = row[:]
                    
        return best_local_row

    # Main SA Loop
    temperature = 100.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if temperature < 0.1:
            break
            
        # Pick a random row
        idx = random.randint(0, n_rows - 1)
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Remove old row's diffs
        remove_row_diffs(old_row, used)
        
        # Try to find a new row
        # Limit the search to slightly better than current global best or current row max
        # To allow moves that increase scope (to escape local minima), we allow max_val_limit to be higher
        # but we prefer lower.
        # Let's set max_val_limit to best_max + 5 (or old_max + 5)
        limit = max(best_max, old_max) + 5
        
        new_row = try_rebuild_row_optimized(idx, used, limit)
        
        if new_row is not None:
            new_max = new_row[-1]
            delta = new_max - old_max
            
            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                # Accept worse moves with probability exp(-delta / temperature)
                if random.random() < pow(sys.e, -delta / temperature):
                    accept = True
                else:
                    accept = False
            
            if accept:
                # Commit change
                add_row_diffs(new_row, used)
                rows[idx] = new_row
                current_max = max(max(r) for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                add_row_diffs(old_row, used)
        else:
            # Revert
            add_row_diffs(old_row, used)
            
        temperature *= cooling_rate
        
        # Periodic restart if stuck? 
        # If no improvement for a while, we could try a more aggressive perturbation.
        # For now, the SA should handle it.
        
    return best_rows
