import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Initialize with greedy
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in rows)
    
    def get_score(rows):
        return max(max(r) for r in rows)
        
    def polish(rows, used_diffs):
        # Modifies rows and used_diffs in place
        changed = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            # Compute current diffs of this row
            diffs_r = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs_r.add(row[j] - row[i])
            
            other_diffs = used_diffs - diffs_r
            
            # Iterate from right to left
            # Note: Changing a_j affects validity of a_{j+1}..a_k
            # But we are just trying to reduce a_j. 
            # If we reduce a_j, it might create a conflict with a_{j+1} (diff change).
            # So we must check against the *current* values of a_{j+1}..a_k.
            # This is what the code below does.
            
            for j in range(k, 0, -1):
                old_val = row[j]
                # Diffs involving old_val
                old_diffs_involved = set()
                for i in range(len(row)):
                    if i != j:
                        old_diffs_involved.add(abs(row[j] - row[i]))
                
                rest_diffs = diffs_r - old_diffs_involved
                
                # Try to find smallest valid v in [row[j-1]+1, old_val-1]
                found = False
                # To speed up, we can limit the search range? 
                # No, we want the smallest.
                for v in range(row[j-1] + 1, old_val):
                    new_diffs_involved = set()
                    for i in range(len(row)):
                        if i != j:
                            new_diffs_involved.add(abs(v - row[i]))
                    
                    # Check if new_diffs_involved conflicts with rest_diffs or other_diffs
                    if not (new_diffs_involved & rest_diffs) and not (new_diffs_involved & other_diffs):
                        # Valid
                        if v != old_val:
                            row[j] = v
                            changed = True
                            # Update used_diffs in place
                            used_diffs.difference_update(old_diffs_involved)
                            used_diffs.update(new_diffs_involved)
                            # Update local diffs_r
                            diffs_r = (diffs_r - old_diffs_involved) | new_diffs_involved
                        found = True
                        break
        return changed

    while time.time() < deadline:
        # 1. Polish all rows
        polish(rows, used_diffs)
        
        current_score = get_score(rows)
        if current_score < best_score:
            best_score = current_score
            best_rows = [r[:] for r in rows]
            
        # 2. Local Move: Row Rebuild
        r_idx = random.randint(0, n_rows - 1)
        old_row = rows[r_idx][:]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        # Remove old row diffs
        used_diffs.difference_update(old_diffs)
        
        # Rebuild row r_idx greedily
        new_row = [0]
        while len(new_row) < k + 1:
            x = new_row[-1] + 1
            while True:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & used_diffs:
                    new_row.append(x)
                    break
                x += 1
        
        # Check if new_row is better or equal
        new_max = new_row[-1]
        old_max = old_row[-1]
        
        # Accept if new_max < old_max OR (new_max == old_max and random() < 0.1)
        # Accepting equal max allows exploration of different diff sets
        if new_max < old_max or (new_max == old_max and random.random() < 0.1):
            rows[r_idx] = new_row
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            used_diffs.update(new_diffs)
        else:
            # Revert
            rows[r_idx] = old_row
            used_diffs.update(old_diffs)

    return best_rows
