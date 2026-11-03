import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # 1. Generate initial solution using Greedy
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
                used |= new_diffs
        rows.append(row)
    
    # Build proper difference tracking: for each row, store all diffs
    # diff_counts[d] = number of times diff d appears across all rows
    diff_counts = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Helper to check if a move is valid
    def try_move(r_idx, j_idx, new_val):
        """Try to set rows[r_idx][j_idx] = new_val. Returns True if valid, updates state if so."""
        nonlocal diff_counts, best_rows, best_max
        
        row = rows[r_idx]
        old_val = row[j_idx]
        
        if old_val == new_val:
            return False
        
        # Check order constraints
        if j_idx > 0 and new_val <= row[j_idx - 1]:
            return False
        if j_idx < k and new_val >= row[j_idx + 1]:
            return False
        
        # Compute removed and added differences
        removed = []
        added = []
        for i in range(len(row)):
            if i == j_idx:
                continue
            old_d = abs(old_val - row[i])
            new_d = abs(new_val - row[i])
            removed.append(old_d)
            added.append(new_d)
        
        # Check if move is valid: for each added diff, if it's not in removed,
        # then diff_counts must be 0 for that diff
        for d in added:
            if d not in removed:
                if diff_counts.get(d, 0) > 0:
                    return False
        
        # Move is valid - apply it
        for d in removed:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in added:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        row[j_idx] = new_val
        
        # Check improvement
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
        
        return True
    
    # 2. Local Search with multiple strategies
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Strategy 1: Try to decrement elements in the row with the largest max
        max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        
        # Try multiple random positions in the max row
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            j_idx = random.randint(1, k)
            old_val = rows[max_row_idx][j_idx]
            
            # Try to find the smallest valid value >= row[j_idx-1]+1
            lower = rows[max_row_idx][j_idx - 1] + 1 if j_idx > 0 else 1
            upper = rows[max_row_idx][j_idx + 1] - 1 if j_idx < k else old_val
            
            # Try a few candidate values
            candidates = set()
            if lower < old_val:
                candidates.add(lower)
                candidates.add(max(lower, old_val - random.randint(1, max(1, old_val // 4))))
                candidates.add(max(lower, old_val - 1))
            
            for new_val in sorted(candidates):
                if new_val < lower or new_val > upper:
                    continue
                if try_move(max_row_idx, j_idx, new_val):
                    break
        
        # Strategy 2: Random row, try to reduce any element
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            old_val = rows[r_idx][j_idx]
            
            lower = rows[r_idx][j_idx - 1] + 1 if j_idx > 0 else 1
            upper = rows[r_idx][j_idx + 1] - 1 if j_idx < k else old_val
            
            if lower < old_val:
                new_val = random.randint(lower, old_val - 1)
                try_move(r_idx, j_idx, new_val)
        
        # Strategy 3: Try to reduce the global max specifically
        # Find the row and position with the maximum value
        global_max = max(max(r) for r in rows)
        for r_idx in range(n_rows):
            if max(rows[r_idx]) < global_max:
                continue
            # This row has the global max at the last position
            j_idx = k
            old_val = rows[r_idx][j_idx]
            lower = rows[r_idx][j_idx - 1] + 1
            
            if lower < old_val:
                # Try to find smallest valid value
                for new_val in range(lower, old_val):
                    if try_move(r_idx, j_idx, new_val):
                        break
                if time.time() - start_time > time_limit:
                    break
    
    return best_rows
