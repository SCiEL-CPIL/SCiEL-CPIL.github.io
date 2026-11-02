"""Difference triangle set construction using simulated annealing with compaction."""
import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_scope = None
    best_rows = None
    
    # Initialize with greedy construction
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
    
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    # Helper: compute all differences for a set of rows
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if rows are valid
    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == n_rows * k * (k+1) // 2
    
    # Helper: compute scope
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Helper: compact rows (shift to minimize max entry)
    def compact_rows(rows):
        # For each row, we can shift it as long as differences remain valid
        # This is complex, so we'll just try to minimize each row individually
        # by ensuring they start at 0 (they already do)
        return rows
    
    # Simulated annealing
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    temperature = 100.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        old_row = list(current_rows[row_idx])
        
        # Try to modify the row: pick a random element and change it
        if len(current_rows[row_idx]) > 1:
            pos = random.randint(1, len(current_rows[row_idx]) - 1)
            old_val = current_rows[row_idx][pos]
            
            # Try nearby values
            for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= current_rows[row_idx][pos-1]:
                    continue
                if pos < len(current_rows[row_idx]) - 1 and new_val >= current_rows[row_idx][pos+1]:
                    continue
                
                # Try this modification
                current_rows[row_idx][pos] = new_val
                
                # Check validity
                if is_valid(current_rows):
                    new_scope = get_scope(current_rows)
                    delta_scope = new_scope - current_scope
                    
                    # Accept if better or with probability based on temperature
                    if delta_scope < 0 or random.random() < pow(2.718281828, -delta_scope / temperature):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]
                        break
                else:
                    current_rows[row_idx][pos] = old_val
                    break
        
        # Occasionally try to rebuild a row from scratch with a different order
        if random.random() < 0.1:
            old_row = list(current_rows[row_idx])
            # Remove this row's differences
            all_diffs = set()
            for i, row in enumerate(current_rows):
                if i != row_idx:
                    for a in range(len(row)):
                        for b in range(a+1, len(row)):
                            all_diffs.add(row[b] - row[a])
            
            # Try to build a new row using greedy with randomization
            new_row, x = [0], 0
            while len(new_row) < k + 1:
                x += 1
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & all_diffs):
                    new_row.append(x)
                    all_diffs |= new_diffs
            
            if new_row != old_row:
                current_rows[row_idx] = new_row
                if is_valid(current_rows):
                    new_scope = get_scope(current_rows)
                    if new_scope < current_scope or random.random() < pow(2.718281828, (new_scope - current_scope) / temperature):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]
        
        temperature *= cooling
    
    return best_rows
