import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Total differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Greedy baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used
    
    rows, used = greedy_construct()
    best_max = max(max(r) for r in rows)
    best_rows = rows
    
    # Try to improve with local search
    # For each row, try to find a better (smaller max) row given the other rows' differences
    
    def get_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    def try_improve_row(row_idx, other_used, max_allowed):
        """Try to find a row with max <= max_allowed that doesn't conflict with other_used."""
        # Backtracking
        row = [0]
        used_in_row = set()
        
        def backtrack(pos, current_max):
            if time.time() > deadline:
                return None
            if pos == k + 1:
                return row[:]
            start_val = row[-1] + 1
            end_val = min(max_allowed, current_max) if pos > 0 else max_allowed
            # Try values in a smart order: prefer smaller gaps for early positions
            for x in range(start_val, end_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & other_used) and not (new_diffs & used_in_row):
                    row.append(x)
                    used_in_row |= new_diffs
                    result = backtrack(pos + 1, max(current_max, x))
                    if result is not None:
                        return result
                    row.pop()
                    used_in_row -= new_diffs
            return None
        
        return backtrack(1, 0)
    
    # Local search: try to reduce each row's max
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            # Compute differences used by all other rows
            other_used = set()
            for i in range(n_rows):
                if i != row_idx:
                    other_used |= get_diffs(rows[i])
            
            current_max = max(rows[row_idx])
            
            # Try to find a row with max < current_max
            target_max = current_max - 1
            if target_max < k:  # minimum possible is k (0,1,2,...,k)
                continue
            
            new_row = try_improve_row(row_idx, other_used, target_max)
            if new_row is not None:
                rows[row_idx] = new_row
                improved = True
                new_max = max(new_row)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
        
        if not improved:
            # Try random restarts of individual rows with different orderings
            for row_idx in range(n_rows):
                if time.time() > deadline:
                    break
                other_used = set()
                for i in range(n_rows):
                    if i != row_idx:
                        other_used |= get_diffs(rows[i])
                
                current_max = max(rows[row_idx])
                # Try a few random target maxes
                for target in range(max(k, current_max - 3), current_max):
                    new_row = try_improve_row(row_idx, other_used, target)
                    if new_row is not None:
                        rows[row_idx] = new_row
                        if max(new_row) < best_max:
                            best_max = max(new_row)
                            best_rows = [r[:] for r in rows]
                        break
    
    return best_rows
