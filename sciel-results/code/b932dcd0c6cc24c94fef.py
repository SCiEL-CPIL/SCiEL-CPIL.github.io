import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with best-fit greedy
    seed = 0
    while time.time() < deadline - 70:  # Reserve 70s for SA + polish
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break
        
        seed += 1
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows

# ... (keep _scope, _all_diffs, _greedy_construct, _packing_score, _simulated_annealing, _is_valid, _generate_candidate_row unchanged) ...

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: targeted local search to reduce the maximum element.
    Uses backtracking to find a valid replacement for the row with the max element.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row diffs
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline - 3:
            break
            
        # Find index of row with max last element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to find a replacement row for max_idx with max < current_max
        forbidden = set()
        for i in range(n_rows):
            if i != max_idx:
                forbidden.update(row_diffs[i])
        
        # Target: find row with max < current_max
        # We use backtracking to search for a row with last element < current_max
        new_row = _solve_row_backtrack(k, forbidden, current_max - 1)
        
        if new_row is not None:
            # Verify validity
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if not (new_diffs & forbidden):
                # Accept if scope improves or stays same (to allow further reductions)
                # But we must not increase scope.
                # Since new_row[-1] < current_max, and current_max was global max,
                # new global max is <= current_max.
                # If there are other rows with max == current_max, scope stays same.
                # If max_idx was unique max, scope drops.
                
                rows[max_idx] = new_row
                row_diffs[max_idx] = new_diffs
                new_scope = _scope(rows)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                elif new_scope == best_scope:
                    # Keep the new row in 'rows' to continue polishing,
                    # but don't update best_rows unless we find a strict improvement later?
                    # Actually, if scope is same, we haven't improved the score yet.
                    # But we might improve it in the next iteration.
                    # So we keep 'rows' updated.
                    pass
                else:
                    # Should not happen
                    pass
        else:
            # Couldn't reduce max_idx
            # Try to reduce other rows?
            # If we can't reduce the max, we might try to reduce a row that is close to max
            # to free up differences, allowing the max row to be reduced later?
            # This is complex. For now, break.
            # Or try random other rows?
            # Let's try to reduce a random other row if we are stuck.
            if _ > 10:
                 # Stuck for a while, try a different row
                 other_idx = random.randint(0, n_rows - 1)
                 if other_idx != max_idx:
                     other_max = rows[other_idx][-1]
                     forbidden_other = set()
                     for i in range(n_rows):
                         if i != other_idx:
                             forbidden_other.update(row_diffs[i])
                     new_row_other = _solve_row_backtrack(k, forbidden_other, other_max - 1)
                     if new_row_other is not None:
                         n = len(new_row_other)
                         new_diffs_other = set()
                         for i in range(n):
                             for j in range(i + 1, n):
                                 new_diffs_other.add(new_row_other[j] - new_row_other[i])
                         if not (new_diffs_other & forbidden_other):
                             rows[other_idx] = new_row_other
                             row_diffs[other_idx] = new_diffs_other
                             # Update scope
                             new_scope = _scope(rows)
                             if new_scope < best_scope:
                                 best_scope = new_scope
                                 best_rows = [list(r) for r in rows]
            else:
                break

    return best_rows

def _solve_row_backtrack(k, forbidden, max_limit):
    """
    Find a row of length k+1 starting with 0, strictly increasing,
    such that all differences are not in 'forbidden' and last element <= max_limit.
    Returns the row or None.
    """
    # Use DFS with MRV-like heuristic: try smaller x first
    # Add a node limit to prevent infinite loops
    node_limit = 50000
    nodes_visited = 0
    
    def dfs(current_row, last_val):
        nonlocal nodes_visited
        if nodes_visited > node_limit:
            return None
        
        if len(current_row) == k + 1:
            return list(current_row)
        
        next_start = last_val + 1
        # Try x from next_start to max_limit
        # Optimization: If we have few slots left, we can bound x?
        # We need k+1 - len(current_row) more elements.
        # The minimum next element is last_val + 1.
        # The maximum next element is max_limit - (slots_remaining - 1)
        slots_remaining = k + 1 - len(current_row)
        max_x = max_limit - (slots_remaining - 1)
        
        for x in range(next_start, max_x + 1):
            nodes_visited += 1
            if nodes_visited > node_limit:
                return None
                
            valid = True
            for a in current_row:
                if x - a in forbidden:
                    valid = False
                    break
            
            if valid:
                current_row.append(x)
                res = dfs(current_row, x)
                if res is not None:
                    return res
                current_row.pop()
                
        return None
        
    return dfs([0], 0)
