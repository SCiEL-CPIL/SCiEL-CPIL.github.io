import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = k + 1  # Number of elements per row
    total_diffs = n_rows * (n * (n - 1) // 2)
    
    # Fast greedy to get initial valid solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Get initial valid solution
    best_rows = greedy_construct()
    best_score = max(max(r) for r in best_rows)
    
    # Data structure for efficient conflict checking
    # diff_counts[d] = number of times difference d appears across all rows
    # We'll use this to quickly determine if adding a row creates conflicts
    
    def compute_diff_counts(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    def get_row_diffs(row):
        """Get set of all positive differences in a row"""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def check_validity(rows):
        """Check if all differences are unique across all rows"""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def find_conflicts(rows):
        """Return list of (d, count) where count > 1"""
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        conflicts = [(d, c) for d, c in counts.items() if c > 1]
        return conflicts
    
    # Local search: try to improve by modifying individual rows
    def try_improve_row(row_idx, current_rows, max_entry_target):
        """Try to find a better row at position row_idx that doesn't conflict with others"""
        # Get differences used by all other rows
        other_diffs = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            for j in range(len(current_rows[i])):
                for l in range(j + 1, len(current_rows[i])):
                    other_diffs.add(current_rows[i][l] - current_rows[i][j])
        
        # Try random restarts with different starting points
        best_row = None
        best_max = max(max(r) for r in current_rows)
        
        for attempt in range(100):
            row = [0]
            x = 0
            valid = True
            while len(row) < n:
                x += 1
                if x > max_entry_target:
                    valid = False
                    break
                new_diffs = {x - a for a in row}
                # Check against other rows' diffs and current row's diffs
                if new_diffs & other_diffs:
                    continue
                # Check for internal conflicts (shouldn't happen if we're building incrementally)
                row.append(x)
            
            if valid and len(row) == n:
                row_max = max(row)
                if row_max < best_max or (best_row is None and row_max < max_entry_target):
                    best_row = row[:]
                    if row_max < best_max:
                        best_max = row_max
                        if best_max <= max_entry_target - 5:
                            break
        
        return best_row
    
    # Main search loop
    current_rows = [row[:] for row in best_rows]
    current_score = best_score
    
    # Strategy: iteratively try to replace each row with a better one
    # Use a simulated annealing-like approach
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.5
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > min_T:
        # Pick a random row to modify
        r_idx = random.randint(0, n_rows - 1)
        old_row = current_rows[r_idx][:]
        old_max = max(old_row)
        
        # Try to find a better row
        new_row = try_improve_row(r_idx, current_rows, current_score)
        
        if new_row is not None:
            new_max = max(new_row)
            
            # Accept if it's strictly better or with some probability if equal
            if new_max < old_max:
                current_rows[r_idx] = new_row
                new_score = max(max(r) for r in current_rows)
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [r[:] for r in current_rows]
                else:
                    no_improve_count += 1
            elif new_max == old_max:
                # Accept with some probability to escape local optima
                if random.random() < 0.3:
                    current_rows[r_idx] = new_row
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling_rate
        
        # If stuck for too long, try a more aggressive restart
        if no_improve_count > max_no_improve:
            # Partial restart: keep some good rows, regenerate others
            # Keep the 3 rows with smallest max values
            row_maxes = [(max(r), i) for i, r in enumerate(current_rows)]
            row_maxes.sort()
            keep_indices = set(idx for _, idx in row_maxes[:3])
            
            # Regenerate other rows
            used_diffs = set()
            for i in range(n_rows):
                if i in keep_indices:
                    for j in range(len(current_rows[i])):
                        for l in range(j + 1, len(current_rows[i])):
                            used_diffs.add(current_rows[i][l] - current_rows[i][j])
            
            for i in range(n_rows):
                if i not in keep_indices:
                    # Try to regenerate this row
                    for attempt in range(50):
                        row = [0]
                        x = 0
                        valid = True
                        while len(row) < n:
                            x += 1
                            if x > best_score + 10:
                                valid = False
                                break
                            new_diffs = {x - a for a in row}
                            if new_diffs & used_diffs:
                                continue
                            row.append(x)
                        
                        if valid and len(row) == n:
                            current_rows[i] = row
                            for j in range(len(row)):
                                for l in range(j + 1, len(row)):
                                    used_diffs.add(row[l] - row[j])
                            break
            
            no_improve_count = 0
            # Reset temperature
            T = max(T, 5.0)
    
    # Final validation
    if not check_validity(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()
    
    return best_rows
