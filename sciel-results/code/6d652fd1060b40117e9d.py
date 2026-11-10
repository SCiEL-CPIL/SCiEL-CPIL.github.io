"""Difference triangle set solver using backtracking with constraint propagation."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Parameters
    num_diffs_per_row = k * (k + 1) // 2  # number of differences per row
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Greedy initial solution to establish a baseline
    best_solution = _greedy_solve(n_rows, k)
    best_scope = max(max(row) for row in best_solution) if best_solution else float('inf')
    
    if not best_solution:
        return []
    
    # Try to improve with backtracking
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Try with the current best scope or slightly less
        target_scope = best_scope - 1 if attempt > 0 else best_scope
        
        if attempt > 0 and attempt % 5 == 0:
            # Occasionally try a tighter bound
            target_scope = max(1, best_scope - 1 - attempt // 5)
        
        solution = _backtrack_solve(n_rows, k, target_scope, deadline)
        if solution is not None:
            scope = max(max(row) for row in solution)
            if scope < best_scope:
                best_scope = scope
                best_solution = solution
            else:
                # Same scope, continue trying to find lower
                continue
    
    return best_solution


def _greedy_solve(n_rows, k):
    """Greedy construction of rows."""
    used = set()
    rows = []
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


def _backtrack_solve(n_rows, k, max_scope, deadline):
    """Backtracking solver that tries to construct all rows within max_scope."""
    
    # Precompute all valid difference sets for a single row
    # A row is defined by its sorted elements [0, a1, a2, ..., ak]
    # The differences are all a_j - a_i for i < j
    
    def get_differences(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Generate candidate rows: all possible rows of length k+1 starting at 0
    # with max element <= max_scope
    # This could be huge, so we use a smarter approach
    
    # Instead, build rows incrementally with backtracking
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row, row_pos):
        if time.time() > deadline - 2.0:
            return False
        
        if row_idx == n_rows:
            return True
        
        if row_pos == k + 1:
            # Row is complete
            rows.append(current_row)
            return backtrack(row_idx + 1, [0], 0)
        
        # Determine the range of possible next values
        if row_pos == 0:
            # Must be 0
            return backtrack(row_idx, current_row + [0], 1)
        
        start = current_row[-1] + 1
        end = min(max_scope, current_row[-1] + max_scope)  # can't exceed scope
        
        # Try values from smallest to largest (greedy preference)
        for val in range(start, end + 1):
            new_row = current_row + [val]
            # Check if adding this value creates any conflict
            new_diffs = set()
            for i in range(len(new_row) - 1):
                diff = val - new_row[i]
                if diff in used_diffs:
                    break
                new_diffs.add(diff)
            else:
                # No conflicts, proceed
                rows_temp = rows[:]
                old_used = used_diffs.copy()
                
                if backtrack(row_idx, new_row, row_pos + 1):
                    return True
                
                # Backtrack
                rows[:] = rows_temp
                used_diffs.clear()
                used_diffs.update(old_used)
        
        return False
    
    # We need to be more careful with the backtracking state
    # Let's use a cleaner approach
    
    def solve():
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row, pos):
            if time.time() > deadline - 2.0:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Complete this row
                rows.append(current_row)
                result = dfs(row_idx + 1, [0], 0)
                if result:
                    return True
                rows.pop()
                return False
            
            if pos == 0:
                return dfs(row_idx, [0], 1)
            
            # Determine valid range for next element
            prev = current_row[-1]
            lo = prev + 1
            hi = min(max_scope, prev + (max_scope - prev))  # effectively max_scope
            
            # Heuristic: try values that minimize new differences conflicts
            candidates = list(range(lo, hi + 1))
            
            # Score candidates by how many conflicts they'd cause
            def conflict_count(val):
                count = 0
                for a in current_row:
                    if (val - a) in used_diffs:
                        count += 1
                return count
            
            # Sort by conflict count, then by value
            candidates.sort(key=lambda v: (conflict_count(v), v))
            
            for val in candidates:
                # Check for conflicts
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = val - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Accept
                used_diffs.update(new_diffs)
                
                if dfs(row_idx, current_row + [val], pos + 1):
                    return True
                
                # Undo
                used_diffs.difference_update(new_diffs)
            
            return False
        
        return dfs(0, [0], 0)
    
    if solve():
        # Reconstruct solution
        # The rows list has been modified during backtracking, 
        # so we need to re-run to capture the final state
        # Actually, let's restructure to return the solution properly
        pass
    
    # The above approach is tricky because of state management
    # Let's use a simpler, more robust approach
    
    def solve2():
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row, pos):
            if time.time() > deadline - 2.0:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if pos == k + 1:
                # Complete this row
                rows.append(current_row)
                result = dfs(row_idx + 1, [0], 0)
                if result is not None:
                    return result
                rows.pop()
                return None
            
            if pos == 0:
                return dfs(row_idx, [0], 1)
            
            prev = current_row[-1]
            lo = prev + 1
            hi = min(max_scope, prev + (max_scope - prev))
            
            candidates = list(range(lo, hi + 1))
            
            def conflict_count(val):
                count = 0
                for a in current_row:
                    if (val - a) in used_diffs:
                        count += 1
                return count
            
            candidates.sort(key=lambda v: (conflict_count(v), v))
            
            for val in candidates:
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = val - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                used_diffs.update(new_diffs)
                
                result = dfs(row_idx, current_row + [val], pos + 1)
                if result is not None:
                    return result
                
                used_diffs.difference_update(new_diffs)
            
            return None
        
        return dfs(0, [0], 0)
    
    return solve2()
