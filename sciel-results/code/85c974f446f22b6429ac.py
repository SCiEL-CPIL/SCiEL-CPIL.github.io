import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Each row has k+1 elements, C(k+1, 2) differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def _valid(rows):
        used = set()
        for row in rows:
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in row_diffs:
                        return False
                    row_diffs.add(d)
            used.update(row_diffs)
        return True
    
    def _diffs_for_row(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _build_all_diffs(rows):
        all_diffs = set()
        for row in rows:
            all_diffs |= _diffs_for_row(row)
        return all_diffs
    
    def _fast_greedy():
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
        return rows
    
    def _sa_optimize(initial_rows, max_time=60.0):
        """Simulated annealing with scope compression move"""
        sa_start = time.time()
        rows = [list(r) for r in initial_rows]
        best_rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        
        # Track used diffs for quick validity checks
        used_diffs = _build_all_diffs(rows)
        
        # Move types: 0 = change a value, 1 = scope compression (reduce max element)
        current_temp = 50.0
        cooling = 0.995
        
        while time.time() - sa_start < max_time and time.time() < deadline:
            current_temp *= cooling
            if current_temp < 0.1:
                current_temp = 0.1
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Scope compression: try to reduce the maximum element
                max_idx = max(range(n_rows), key=lambda i: max(rows[i]))
                max_val = max(rows[max_idx])
                
                # Find which position has the max value
                pos = rows[max_idx].index(max_val)
                
                # Try to reduce it
                new_val = max_val - random.randint(1, 3)
                if new_val <= rows[max_idx][pos - 1] if pos > 0 else False:
                    continue
                
                # Check if we can replace
                old_diffs = _diffs_for_row(rows[max_idx])
                rows[max_idx][pos] = new_val
                new_diffs = _diffs_for_row(rows[max_idx])
                
                # Check validity
                all_diffs = used_diffs - old_diffs | new_diffs
                if len(all_diffs) == total_diffs:
                    scope = _scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                        used_diffs = all_diffs
                else:
                    rows[max_idx][pos] = max_val
            else:
                # Standard move: change a random element
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)  # Don't change first element (always 0)
                
                old_val = rows[row_idx][pos]
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                
                # Ensure strictly increasing
                if pos > 0 and new_val <= rows[row_idx][pos - 1]:
                    continue
                if pos < k and new_val >= rows[row_idx][pos + 1]:
                    continue
                if new_val < 0:
                    continue
                
                # Check if change is valid
                old_diffs = _diffs_for_row(rows[row_idx])
                rows[row_idx][pos] = new_val
                new_diffs = _diffs_for_row(rows[row_idx])
                
                all_diffs = used_diffs - old_diffs | new_diffs
                valid = len(all_diffs) == total_diffs
                
                if valid:
                    scope = _scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    used_diffs = all_diffs
                else:
                    # Accept with probability based on temperature
                    rows[row_idx][pos] = old_val
        
        return best_rows, best_scope
    
    # Get initial greedy solution
    best_result = _fast_greedy()
    best_scope = _scope(best_result)
    
    # Run simulated annealing
    if time.time() < deadline:
        best_result, best_scope = _sa_optimize(best_result, max_time=90.0)
    
    # Low-temperature polish pass
    if time.time() < deadline:
        # Try to further reduce scope with more aggressive moves
        polish_start = time.time()
        polish_deadline = min(deadline, polish_start + 15.0)
        
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            
            # Find the row with the largest element
            max_row_idx = max(range(n_rows), key=lambda i: max(best_result[i]))
            max_val = max(best_result[max_row_idx])
            
            # Try to reduce the maximum element
            pos = best_result[max_row_idx].index(max_val)
            
            for new_val in range(max_val - 1, best_result[max_row_idx][pos - 1] if pos > 0 else -1, -1):
                if new_val <= (best_result[max_row_idx][pos - 1] if pos > 0 else -1):
                    break
                
                old_diffs = _diffs_for_row(best_result[max_row_idx])
                best_result[max_row_idx][pos] = new_val
                new_diffs = _diffs_for_row(best_result[max_row_idx])
                
                all_diffs = _build_all_diffs(best_result)
                if len(all_diffs) == total_diffs:
                    new_scope = _scope(best_result)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    break
                else:
                    best_result[max_row_idx][pos] = max_val
        
        # Final validation
        if not _valid(best_result):
            best_result = _fast_greedy()
            best_scope = _scope(best_result)
    
    return best_result
