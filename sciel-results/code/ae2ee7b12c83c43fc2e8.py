import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin

    # Calculate lower bound
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Start from lower bound
    # If lower bound is too low, we will just fail fast and increment
    start_max_val = lower_bound
    
    # Fallback solution
    best_solution = _greedy_construct(n_rows, k)
    best_max_val = max(max(row) for row in best_solution)
    
    # Search for better solution
    # We try to find a solution with max_val < best_max_val
    # We iterate max_val from lower_bound upwards
    # We stop if we find a solution, or if we exceed best_max_val (since we already have a valid one)
    # Actually, we want to minimize max_val.
    # So we iterate m from lower_bound to best_max_val - 1.
    # If we find a solution, we update best_solution and best_max_val.
    # We can break early if we find a solution at the lower bound?
    # No, we want the *smallest* max_val.
    # So if we find a solution for m, we return it immediately?
    # Yes, because we are iterating m upwards. The first valid m is the minimum.
    
    # However, backtracking might be slow for small m if no solution exists.
    # We should limit the time spent on each m.
    
    for m in range(start_max_val, best_max_val):
        if time.time() > deadline:
            break
            
        # Try to find a solution with max_val = m
        # Use randomized backtracking with time limit
        solution = _backtrack_search(n_rows, k, m, start_time, deadline)
        if solution is not None:
            return solution
            
    return best_solution

def _backtrack_search(n_rows, k, max_val, start_time, deadline):
    """
    Randomized backtracking to find n_rows disjoint rows of size k+1 
    starting with 0, with max element <= max_val.
    Returns list of rows or None.
    """
    used_diffs = set()
    rows = []
    diffs_per_row = (k + 1) * k // 2
    
    def backtrack(row_idx, current_row):
        nonlocal used_diffs, rows
        
        if row_idx == n_rows:
            return True
        
        # Pruning: Check if enough differences are available
        # Diffs used so far: len(used_diffs)
        # Diffs available in 1..max_val: max_val
        # Diffs needed:
        # 1. Complete current row
        # 2. All remaining rows
        
        L = len(current_row)
        # Diffs already in current row: L*(L-1)//2
        # Diffs total in full row: diffs_per_row
        # Diffs needed for current row completion: diffs_per_row - L*(L-1)//2
        # Diffs needed for remaining rows: (n_rows - row_idx - 1) * diffs_per_row
        # Note: If current row is finished (L == k+1), it is already appended to rows?
        # In my logic, I append to rows when L == k+1.
        # So if L == k+1, we are starting a new row?
        # Let's restructure:
        # If L == k+1, we have completed a row.
        # We should have appended it and moved to next row_idx.
        # So in this function, L < k+1 always, except maybe at the start of a new row?
        # No, if L == k+1, we recurse with row_idx+1 and [0].
        # So here, L is always < k+1.
        
        diffs_needed_current = diffs_per_row - (L * (L - 1) // 2)
        diffs_needed_remaining = (n_rows - row_idx - 1) * diffs_per_row
        total_diffs_needed = diffs_needed_current + diffs_needed_remaining
        
        if max_val - len(used_diffs) < total_diffs_needed:
            return False
            
        last = current_row[-1]
        
        # Candidates
        candidates = []
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            candidates.append(x)
            
        if not candidates:
            return False
            
        # Shuffle
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                return False
                
            new_diffs = {x - a for a in current_row}
            current_row.append(x)
            used_diffs |= new_diffs
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
            else:
                if backtrack(row_idx, current_row):
                    return True
            
            current_row.pop()
            used_diffs -= new_diffs
            
        return False

    # Start
    return backtrack(0, [0])

def _greedy_construct(n_rows, k):
    """
    Fallback greedy construction.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    return rows
