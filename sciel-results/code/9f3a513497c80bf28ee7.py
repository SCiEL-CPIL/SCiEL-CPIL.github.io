import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Initial Solution
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_rows)
    
    # 2. Optimization
    # We use a loop that tries to improve the scope.
    # To escape local minima, we use random restarts of the optimization process
    # starting from the current best solution, with randomized row processing order.
    
    attempts = 0
    max_attempts = 1000 # Safety cap
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        current_rows = [row[:] for row in best_rows]
        
        # Randomize row order
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        # Optimization Pass
        # We repeat until no single element can be reduced
        while True:
            pass_improved = False
            for r_idx in row_order:
                row = current_rows[r_idx]
                # Process from right to left
                for j in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        return best_rows
                    
                    current_val = row[j]
                    min_val = row[j-1] + 1
                    
                    if current_val == min_val:
                        continue # Cannot reduce further
                    
                    # Build used differences set excluding those involving row[j]
                    used_diffs = set()
                    for r in current_rows:
                        for i in range(len(r)):
                            for l in range(i+1, len(r)):
                                used_diffs.add(r[l] - r[i])
                    
                    # Remove diffs involving row[j] in row r_idx
                    for i in range(j):
                        used_diffs.remove(row[j] - row[i])
                    for l in range(j+1, len(row)):
                        used_diffs.remove(row[l] - row[j])
                    
                    # Try to find a smaller value
                    # We iterate from min_val to current_val - 1
                    # To speed up, we can check if any candidate is valid.
                    # We want the smallest valid one.
                    
                    # Optimization: If current_val is small, range is small.
                    # If current_val is large, range is large.
                    
                    new_val_found = None
                    for new_val in range(min_val, current_val):
                        # Check validity
                        conflict = False
                        # Check left diffs
                        for i in range(j):
                            d = new_val - row[i]
                            if d in used_diffs:
                                conflict = True
                                break
                        if conflict:
                            continue
                        # Check right diffs
                        for l in range(j+1, len(row)):
                            d = row[l] - new_val
                            if d in used_diffs:
                                conflict = True
                                break
                        if conflict:
                            continue
                        # Check cross diffs (left vs right)
                        # Left diffs: {new_val - row[i] for i < j}
                        # Right diffs: {row[l] - new_val for l > j}
                        # We need to ensure no d_left == d_right
                        # This is equivalent to checking if the set of all new diffs has duplicates.
                        # Since we checked against used_diffs, we just need to check internal consistency.
                        
                        # Actually, if new_val - row[i] == row[l] - new_val, then 2*new_val = row[i] + row[l].
                        # We can just generate the set and check length.
                        
                        left_diffs = {new_val - row[i] for i in range(j)}
                        right_diffs = {row[l] - new_val for l in range(j+1, len(row))}
                        
                        if len(left_diffs) != j or len(right_diffs) != len(row) - j - 1:
                             # This shouldn't happen if row is strictly increasing
                             pass
                             
                        if left_diffs & right_diffs:
                             conflict = True
                             continue
                             
                        if left_diffs & used_diffs or right_diffs & used_diffs:
                             conflict = True
                             continue
                             
                        if not conflict:
                            new_val_found = new_val
                            break # Found smallest valid
                    
                    if new_val_found is not None:
                        row[j] = new_val_found
                        pass_improved = True
                        # We don't update used_diffs here, we rebuild it next time
                        # This is inefficient but simple and fast enough for N=4, K=7.
                        
            if not pass_improved:
                break
                
        # Check if we improved the global scope
        current_scope = max(max(row) for row in current_rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [row[:] for row in current_rows]
            
    return best_rows

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
