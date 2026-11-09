import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_max_scope(rows):
        return max(max(r) for r in rows)
    
    # Backtracking search that builds rows one at a time
    # Key insight: we can try to fit each new row into the remaining unused differences
    def backtrack_search(used_diffs, rows, row_idx, max_val_bound):
        """Try to build row row_idx given used_diffs. Returns row or None."""
        if row_idx >= n_rows:
            return rows
        
        # Try different starting strategies for this row
        # Strategy: try to pack the row tightly
        row = [0]
        remaining_positions = k
        
        # Try greedy with backtracking for this single row
        def build_row_partial(row, used_diffs, pos):
            if pos == k + 1:
                return row[:]
            
            # Try values from low to high
            low = row[-1] + 1 if row else 1
            high = max_val_bound
            
            for val in range(low, high + 1):
                # Check if this value creates valid differences
                new_diffs = set()
                valid = True
                for i in range(len(row)):
                    d = val - row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Also need to check: can we still complete the row?
                # Simple heuristic: the remaining positions need at least
                # (k+1-pos-1) more values, each needing at least 1 more than previous
                # So we need room: (k+1-pos-1) values each at least 1 apart
                # Minimum additional space = k+1-pos-1
                # But also each new value creates differences with all existing values
                # This is complex, so just try it
                
                row.append(val)
                used_diffs.update(new_diffs)
                
                result = build_row_partial(row, used_diffs, pos + 1)
                if result is not None:
                    return result
                
                # Backtrack
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        result = build_row_partial(row, used_diffs, 1)
        if result is not None:
            rows.append(result)
            return backtrack_search(used_diffs, rows, row_idx + 1, max_val_bound)
        
        return None
    
    # Better: use a more structured approach with constraint propagation
    # Represent the problem as: for each row, we have k+1 values a_0=0 < a_1 < ... < a_k
    # The differences a_j - a_i for all i<j must be unique across all rows
    
    def solve_with_backtracking(target_scope):
        """Try to find a solution with max scope <= target_scope."""
        used_diffs = set()
        rows = []
        
        def try_build_row(row_idx, current_max):
            if row_idx == n_rows:
                return True
            
            # Try to build row row_idx
            # Use DFS with constraint propagation
            row = [0]
            remaining = k
            
            def dfs(pos):
                if pos == k + 1:
                    rows.append(row[:])
                    return try_build_row(row_idx + 1, max(current_max, row[-1]))
                
                low = row[-1] + 1
                high = min(target_scope, current_max + remaining) if remaining > 0 else target_scope
                
                # Try values, prefer smaller ones to keep scope tight
                for val in range(low, high + 1):
                    # Check differences with existing elements in this row
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if conflict:
                        continue
                    
                    # Pruning: check if we can still fit remaining elements
                    # Each remaining element must be > val, and each creates new diffs
                    # Simple bound: need at least (k+1-pos-1) more values, each at least 1 apart
                    # So max needed = val + (k+1-pos-1)
                    # But also each new value creates diffs with all previous, so more constraints
                    # For now, just check basic feasibility
                    if val + (k + 1 - pos - 1) > target_scope:
                        continue
                    
                    row.append(val)
                    used_diffs.update(new_diffs)
                    
                    if dfs(pos + 1):
                        return True
                    
                    row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            if dfs(1):
                return True
            
            return False
        
        return try_build_row(0, 0)
    
    # First, get a greedy baseline
    def generate_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs
    
    # Get initial greedy solution
    best_rows, best_used_diffs = generate_greedy()
    best_scope = get_max_scope(best_rows)
    
    # Try to improve using backtracking with decreasing scope bounds
    # Start from best_scope and try to go lower
    for trial_scope in range(best_scope, best_scope - 20, -1):
        if time.time() > deadline - 30:
            break
        
        # Reset and try to solve with this scope bound
        success = solve_with_backtracking(trial_scope)
        
        if success:
            new_scope = get_max_scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                # Continue trying to go lower
            else:
                break
    
    # If backtracking didn't improve, try hill climbing on the best solution
    if time.time() < deadline - 20:
        def get_involved_diffs(row, pos, val):
            diffs = set()
            for i in range(pos):
                diffs.add(val - row[i])
            for j in range(pos + 1, len(row)):
                diffs.add(row[j] - val)
            return diffs
        
        rows = [r[:] for r in best_rows]
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_diffs.add(r[j] - r[i])
        current_scope = get_max_scope(rows)
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for r_idx in range(n_rows):
                if time.time() > deadline - 10:
                    break
                row = rows[r_idx]
                for pos in range(1, k + 1):
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < k else old_val
                    
                    for new_val in range(low, old_val):
                        old_involved = get_involved_diffs(row, pos, old_val)
                        used_diffs.difference_update(old_involved)
                        new_involved = get_involved_diffs(row, pos, new_val)
                        
                        if len(new_involved) != len(old_involved):
                            used_diffs |= old_involved
                            continue
                        if new_involved & used_diffs:
                            used_diffs |= old_involved
                            continue
                        
                        used_diffs |= new_involved
                        row[pos] = new_val
                        new_scope = get_max_scope(rows)
                        
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                        else:
                            used_diffs.difference_update(new_involved)
                            used_diffs |= old_involved
                            row[pos] = old_val
                    
                    if improved:
                        break
                if improved:
                    break
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
    
    return best_rows
