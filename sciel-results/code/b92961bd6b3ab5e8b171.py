import random
import math
import time

def construct(n_rows, k):
    # Bin-packing heuristic for difference space
    # Strategy: Assign differences to rows to minimize maximum span
    
    # Step 1: Greedy initial construction
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
    
    current_max = max(max(row) for row in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Helper functions
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_row(row):
        if len(row) != len(set(row)):
            return False
        if any(row[i] >= row[i+1] for i in range(len(row)-1)):
            return False
        diffs = get_row_diffs(row)
        return len(diffs) == len(row)*(len(row)-1)//2
    
    def check_and_apply_move(rows, row_idx, point_idx, new_val, used_set):
        """Check if move is valid and apply it, returning True if successful."""
        row = rows[row_idx]
        old_val = row[point_idx]
        
        if point_idx > 0 and new_val <= row[point_idx-1]:
            return False
        if point_idx < len(row)-1 and new_val >= row[point_idx+1]:
            return False
        
        # Temporarily apply
        row[point_idx] = new_val
        
        if not is_valid_row(row):
            row[point_idx] = old_val
            return False
        
        old_row = row[:]
        old_row[point_idx] = old_val
        old_diffs = get_row_diffs(old_row)
        new_diffs = get_row_diffs(row)
        
        removed = old_diffs - new_diffs
        added = new_diffs - old_diffs
        
        for d in added:
            if d in used_set and d not in removed:
                row[point_idx] = old_val
                return False
        
        # Apply the move
        used_set -= removed
        used_set |= added
        return True
    
    # Local search with systematic improvement
    start_time = time.time()
    max_time = 110
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try to improve rows in order of their span
        for row_idx in sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True):
            row = rows[row_idx]
            
            # Try moving the last point to reduce span
            if row[-1] > 0:
                min_val = row[-2] + 1 if len(row) > 1 else 1
                for new_val in range(min_val, row[-1]):
                    if check_and_apply_move(rows, row_idx, len(row)-1, new_val, used):
                        improved = True
                        break
            
            # If still large, try other points
            if row[-1] > best_max * 0.9:
                for point_idx in range(len(row)-2, 0, -1):
                    current_val = row[point_idx]
                    min_val = row[point_idx-1] + 1
                    if current_val > min_val + 2:
                        # Try a few candidate values
                        candidates = [min_val, min_val + 1, (min_val + current_val)//2]
                        random.shuffle(candidates)
                        for new_val in candidates:
                            if new_val < current_val and check_and_apply_move(rows, row_idx, point_idx, new_val, used):
                                improved = True
                                break
                        if improved:
                            break
            
            if improved:
                break
        
        # If no systematic improvement, try random moves
        if not improved:
            for _ in range(10):
                row_idx = random.randint(0, n_rows - 1)
                point_idx = random.randint(1, len(rows[row_idx]) - 1)
                current_val = rows[row_idx][point_idx]
                
                if current_val > 2:
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = current_val + delta
                    if new_val > 0:
                        if check_and_apply_move(rows, row_idx, point_idx, new_val, used):
                            improved = True
                            break
        
        # Update best
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early exit if stuck for too long
        if no_improve_count > 5000:
            break
    
    return best_rows
