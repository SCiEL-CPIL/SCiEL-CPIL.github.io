"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def greedy_construct(n_rows, k, used_initial=None, rows_initial=None):
        """Build rows greedily, optionally starting from initial state."""
        used = set(used_initial) if used_initial else set()
        rows = list(rows_initial) if rows_initial else []
        
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows, used
    
    def get_used(rows):
        """Compute the set of all used differences from rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for row in rows:
            for d in row_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True
    
    def scope(rows):
        """Return the maximum entry across all rows."""
        return max(max(row) for row in rows)
    
    # Baseline greedy
    rows, used = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = scope(best_rows)
    
    if time_left() < 5:
        return best_rows
    
    # Local search: try to reduce scope by modifying individual rows
    def try_reduce_row(rows, row_idx, used_other, max_attempts=1000):
        """Try to find a better row at row_idx with smaller max entry."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        
        # Try greedy with different starting strategies
        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            # Try a perturbed greedy
            row, x = [0], 0
            success = True
            while len(row) < k + 1:
                # Try several candidates
                found = False
                for trial_x in range(x+1, x+100):
                    new = {trial_x - a for a in row}
                    if not new & used_other and not (new & set()) :
                        # Check against used_other
                        if not (new & used_other):
                            row.append(trial_x)
                            found = True
                            break
                if not found:
                    success = False
                    break
                x = row[-1]
            
            if success and max(row) < best_max:
                # Verify full validity
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        
        return best_row, best_max
    
    # Try to improve each row
    improved = True
    while improved and time_left() > 10:
        improved = False
        for i in range(n_rows):
            if time_left() < 10:
                break
            
            # Compute used differences from other rows
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= row_diffs(best_rows[j])
            
            new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=200)
            
            if new_max < max(best_rows[i]):
                old_scope = scope(best_rows)
                best_rows[i] = new_row
                new_scope = scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
    
    # Try random restarts with different random seeds
    if time_left() > 30:
        for restart in range(20):
            if time_left() < 15:
                break
            
            random.seed(restart * 12345 + 67890)
            
            # Perturbed greedy: sometimes skip a candidate to explore different paths
            used = set()
            rows = []
            success = True
            for r_idx in range(n_rows):
                row, x = [0], 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        # With some probability, skip this candidate to explore alternatives
                        if random.random() < 0.1 and x < 200:
                            continue
                        row.append(x)
                        used |= new
                if len(row) < k + 1:
                    success = False
                    break
                rows.append(row)
            
            if success:
                s = scope(rows)
                if s < best_scope:
                    best_rows = [list(r) for r in rows]
                    best_scope = s
    
    return best_rows
# EVOLVE-BLOCK-END
