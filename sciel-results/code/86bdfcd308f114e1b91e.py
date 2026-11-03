import random
import math
import time

def construct(n_rows, k):
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Helper: Generate a greedy row given forbidden diffs
    def greedy_row(forbidden_diffs):
        row = [0]
        used = set(forbidden_diffs)
        x = 1
        while len(row) < k + 1:
            # Check if x is valid
            # Differences are x - a for a in row
            # If any of these are in used, x is invalid.
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]: # Update used with new diffs
                    used.add(x - a)
                x += 1
            else:
                x += 1
        return row

    # Initial Solution
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        r = greedy_row(used_diffs)
        rows.append(r)
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used_diffs.add(r[j] - r[i])
    
    def get_max_val(rows_list):
        m = 0
        for r in rows_list:
            if r[-1] > m:
                m = r[-1]
        return m

    current_cost = get_max_val(rows)
    best_rows = [r[:] for r in rows]
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 110
    
    T = 50.0
    cooling = 0.9995
    min_T = 1e-6
    
    # Pre-compute diffs for all rows to make removal/addition faster?
    # No, k is small, re-computing is fine.
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Select two distinct rows to re-optimize
        if n_rows < 2:
            break
            
        r1_idx = random.randint(0, n_rows - 1)
        r2_idx = random.randint(0, n_rows - 1)
        if r1_idx == r2_idx:
            continue
            
        old_row1 = rows[r1_idx]
        old_row2 = rows[r2_idx]
        
        # Remove diffs of these two rows from global used_diffs
        for i in range(len(old_row1)):
            for j in range(i+1, len(old_row1)):
                used_diffs.discard(old_row1[j] - old_row1[i])
        for i in range(len(old_row2)):
            for j in range(i+1, len(old_row2)):
                used_diffs.discard(old_row2[j] - old_row2[i])
        
        # Determine build order
        build_1_first = random.random() < 0.5
        
        new_row1 = None
        new_row2 = None
        
        if build_1_first:
            # Build row 1 against current used_diffs (which excludes 1 and 2)
            new_row1 = greedy_row(used_diffs)
            
            # Add row 1 diffs to used_diffs temporarily
            diff1 = set()
            for i in range(len(new_row1)):
                for j in range(i+1, len(new_row1)):
                    diff1.add(new_row1[j] - new_row1[i])
            used_diffs.update(diff1)
            
            # Build row 2 against used_diffs (now includes new row 1)
            new_row2 = greedy_row(used_diffs)
            
            # Calculate new max
            other_max = 0
            for i in range(n_rows):
                if i != r1_idx and i != r2_idx:
                    if rows[i][-1] > other_max:
                        other_max = rows[i][-1]
            
            new_cost = max(new_row1[-1], new_row2[-1], other_max)
            
        else:
            # Build row 2 first
            new_row2 = greedy_row(used_diffs)
            
            diff2 = set()
            for i in range(len(new_row2)):
                for j in range(i+1, len(new_row2)):
                    diff2.add(new_row2[j] - new_row2[i])
            used_diffs.update(diff2)
            
            # Build row 1
            new_row1 = greedy_row(used_diffs)
            
            other_max = 0
            for i in range(n_rows):
                if i != r1_idx and i != r2_idx:
                    if rows[i][-1] > other_max:
                        other_max = rows[i][-1]
            
            new_cost = max(new_row1[-1], new_row2[-1], other_max)
        
        # SA Decision
        delta = new_cost - current_cost
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta / T)
        
        if accept:
            # Commit changes
            rows[r1_idx] = new_row1
            rows[r2_idx] = new_row2
            current_cost = new_cost
            
            # Update used_diffs with the NEW diffs of the accepted rows
            # Note: used_diffs currently contains diffs of new_row1 (if built first) or new_row2 (if built first)
            # It does NOT contain the diffs of the second row built.
            # And it does NOT contain the old diffs (which were removed).
            
            # We need to add the diffs of the row that was built SECOND.
            if build_1_first:
                # new_row2 diffs are not in used_diffs yet
                for i in range(len(new_row2)):
                    for j in range(i+1, len(new_row2)):
                        used_diffs.add(new_row2[j] - new_row2[i])
            else:
                # new_row1 diffs are not in used_diffs yet
                for i in range(len(new_row1)):
                    for j in range(i+1, len(new_row1)):
                        used_diffs.add(new_row1[j] - new_row1[i])
                        
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in rows]
        else:
            # Reject: Restore old state
            # used_diffs is currently missing old_row1 and old_row2 diffs.
            # It might contain partial new diffs (from the first built row).
            # We need to clean up and restore.
            
            # Easiest way: Rebuild used_diffs from scratch? O(n*k^2). n is small.
            # Or, just add back the old diffs and remove the new partial diffs?
            # Let's just rebuild for safety and simplicity, as n_rows is likely small (e.g., 4, 5, 10).
            used_diffs.clear()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        used_diffs.add(r[j] - r[i])
            
        T *= cooling

    return best_rows
