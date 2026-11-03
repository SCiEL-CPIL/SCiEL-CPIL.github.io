import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # max entry >= num_diffs - 1 (differences are 1..M)
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Bitset-based backtracking search with symmetry breaking
    if time.time() < deadline - 10.0:
        # Try to find better solutions with randomized backtracking
        attempt = 0
        while time.time() < deadline - 5.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
            
            # Try multiple random restarts for this target scope
            found = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
                if solution:
                    new_scope = max(max(row) for row in solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = solution
                        found = True
                        # Try to improve further
                        improved = _improve_scope(best_solution, n_rows, k, deadline)
                        new_scope = max(max(row) for row in improved)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_solution = improved
                        if best_scope <= lower_bound:
                            return best_solution
                        break
                attempt += 1
            
            if not found:
                # If we can't find anything, try a slightly larger target
                if target_scope <= lower_bound + 5:
                    break
                target_scope = min(best_scope - 1, target_scope + 2)
    
    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Bitset-based backtracking search with symmetry breaking.
    Uses a bitmask to track used differences for fast lookup.
    Enforces row ordering symmetry breaking: rows[i][1] > rows[i-1][1]
    """
    rng = random.Random(seed_offset * 7919 + 4242)
    k_plus_1 = k + 1
    
    # Bitmask for used differences (0..max_scope)
    # We'll use a Python int as a bitset
    used_mask = 0
    
    rows = []
    prev_first_elem = 0  # For symmetry breaking
    
    def build_row(pos, current_mark, row_so_far, min_first_elem):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Prune upper bound
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
        
        # For the first element (pos=1), enforce symmetry breaking
        if pos == 1:
            start_x = max(current_mark + 1, min_first_elem)
        else:
            start_x = current_mark + 1
        
        valid_candidates = []
        for x in range(start_x, max_x + 1):
            # Check if all new differences are unused using bitmask
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize with bias towards smaller values
        n_smallest = min(15, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Compute bitmask of new differences
            new_diff_mask = 0
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    break
                new_diff_mask |= (1 << d)
            
            # Check for conflicts with current used_mask
            if new_diff_mask & used_mask:
                continue
            
            # Apply change
            old_used = used_mask
            used_mask_new = used_mask | new_diff_mask
            
            res = build_row(pos + 1, x, row_so_far + [x], min_first_elem)
            if res is not None:
                return res
        
        return None
    
    for i in range(n_rows):
        # Symmetry breaking: first element of row i must be > first element of row i-1
        if i == 0:
            min_first = 1
        else:
            min_first = rows[i-1][1] + 1
        
        row = build_row(1, 0, [0], min_first)
        if row is None:
            return None
        rows.append(row)
        
        # Update used_mask with all differences in this row
        for a_idx in range(k_plus_1):
            for b_idx in range(a_idx + 1, k_plus_1):
                d = row[b_idx] - row[a_idx]
                used_mask |= (1 << d)
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    improved = True
    max_iterations = 20
    iteration = 0
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Only try to shift if it could reduce max or is the max
                for new_val in range(row[pos - 1] + 1, original):
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
