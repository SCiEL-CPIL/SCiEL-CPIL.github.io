"""Difference triangle set: local search with simulated annealing."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Step 1: Greedy construction to get a valid starting solution
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
    
    # Track the current best solution
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Helper: compute all differences for a set of rows
    def compute_used_diffs(rls):
        s = set()
        for row in rls:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    s.add(row[j] - row[i])
        return s
    
    # Helper: check if a row is internally consistent (strictly increasing, starts at 0)
    def valid_row(row):
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True
    
    # Helper: try to reduce the max entry of a specific row by local moves
    # We try to "compress" the row by reducing the last element and adjusting
    def try_compress_row(rows, row_idx, max_allowed, used_diffs):
        """Try to reduce the maximum entry of rows[row_idx] to be <= max_allowed."""
        row = rows[row_idx][:]
        if max(row) <= max_allowed:
            return None
        
        # Remove this row's differences from the global set
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                row_diffs.add(row[j] - row[i])
        
        # Try to find a compressed version of this row
        # Strategy: keep the first few elements, try to reduce the tail
        best_row = None
        best_max_in_row = max(row)
        
        # Try different strategies for compression
        # Strategy 1: Reduce the last element
        # Strategy 2: Reduce multiple tail elements
        
        # We'll use a simple approach: try to find a valid row with max <= max_allowed
        # that doesn't conflict with other rows' differences
        
        other_diffs = used_diffs - row_diffs
        
        # Try to reconstruct the row with a smaller maximum
        # Use a greedy approach with the constraint that max <= max_allowed
        for attempt in range(200):
            new_row = [0]
            x = 0
            success = True
            for pos in range(1, k+1):
                # Try values from current x+1 up to max_allowed
                found = False
                # For the last position, we must stay <= max_allowed
                upper = max_allowed if pos == k else max_allowed
                for val in range(x + 1, upper + 1):
                    new_diffs = {val - a for a in new_row}
                    if not (new_diffs & other_diffs) and not (new_diffs & set()):
                        # Also check against already-placed elements in this new_row
                        # (new_diffs already covers this)
                        new_row.append(val)
                        x = val
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success and max(new_row) < best_max_in_row:
                best_row = new_row
                best_max_in_row = max(new_row)
        
        if best_row is not None:
            return best_row
        return None
    
    # Main local search loop
    # We'll try to iteratively reduce the maximum entry
    current_max = best_max
    
    # Simulated annealing-like approach:
    # Try to reduce the global maximum by compressing the row that has the max
    
    iteration = 0
    stuck_count = 0
    
    while time.time() < deadline:
        iteration += 1
        stuck_count += 1
        
        # Find the row with the maximum entry
        max_val = max(max(r) for r in rows)
        if max_val <= best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows]
        
        if max_val <= 105:  # counting bound, can't do better
            break
        
        # Pick a row to try to compress (the one with the largest entry)
        target_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        target_max = max(rows[target_idx])
        
        # Try to compress this row to have max < target_max
        # Remove its diffs
        row_diffs = set()
        for i in range(len(rows[target_idx])):
            for j in range(i+1, len(rows[target_idx])):
                row_diffs.add(rows[target_idx][j] - rows[target_idx][i])
        
        other_diffs = set()
        for ri in range(n_rows):
            if ri == target_idx:
                continue
            for i in range(len(rows[ri])):
                for j in range(i+1, len(rows[ri])):
                    other_diffs.add(rows[ri][j] - rows[ri][i])
        
        # Try to find a better version of this row with max < target_max
        improved = False
        # Try a range of target maxima
        for new_target in range(target_max - 1, max(target_max - 30, 50), -1):
            new_row = [0]
            x = 0
            success = True
            for pos in range(1, k+1):
                found = False
                upper = new_target
                for val in range(x + 1, upper + 1):
                    new_diffs = {val - a for a in new_row}
                    if not (new_diffs & other_diffs):
                        new_row.append(val)
                        x = val
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                rows[target_idx] = new_row
                improved = True
                break
        
        if improved:
            stuck_count = 0
        else:
            # Try a random perturbation: swap two rows' assignments or shuffle
            if stuck_count > 50:
                # Try random restart with different greedy order
                random.shuffle(rows)
                used = set()
                new_rows = []
                for _ in range(n_rows):
                    row, x = [0], 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                    new_rows.append(row)
                # Only accept if better
                new_max = max(max(r) for r in new_rows)
                if new_max < best_max:
                    rows = new_rows
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
                stuck_count = 0
        
        # Check deadline
        if time.time() > deadline - 5:
            break
    
    # Return the best solution found
    return best_rows
# EVOLVE-BLOCK-END
