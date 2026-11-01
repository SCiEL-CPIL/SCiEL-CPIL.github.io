import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def greedy_solution():
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(n_rows * k):
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
            if target_row_idx == -1:
                break
            i = target_row_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            while True:
                if is_valid_extension(current_row, x, used):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used.add(x - v)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    baseline_rows, baseline_cost = greedy_solution()
    best_solution = baseline_rows
    best_cost = baseline_cost
    
    beam_width = 2000 
    max_depth = n_rows * row_len
    
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    beam = [(0, initial_rows, initial_used)]
    
    # We expand level by level.
    # A "move" is adding one point to one row.
    # Total moves: n_rows * k.
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            # Identify the row to expand
            # Heuristic: Expand the row with the smallest length.
            # Tie-breaker: Smallest last element (to keep values low).
            # If all rows are full, this state is complete (should have been caught earlier).
            
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
                        
            if target_row_idx == -1:
                # All rows full, check if it's a better solution
                current_max = max(max(r) for r in rows)
                if current_max < best_cost:
                    best_cost = current_max
                    best_solution = [r[:] for r in rows]
                continue

            i = target_row_idx
            current_row = rows[i]
            
            # Find valid candidates
            # Use best_cost as upper bound
            limit_val = best_cost - 1
            
            candidates = []
            x = current_row[-1] + 1
            while x <= limit_val:
                if is_valid_extension(current_row, x, used_diffs):
                    candidates.append(x)
                    if len(candidates) >= 20:
                        break
                x += 1
                
            if not candidates:
                # If no candidates in window, try a wider window?
                # Or just skip this state (it's a dead end for this specific expansion choice)
                # Since we force a specific row, this state might be a dead end.
                continue

            # Create new states for each candidate
            for new_val in candidates:
                new_rows = [r[:] for r in rows]
                new_rows[i].append(new_val)
                
                # Update used_diffs
                new_used_diffs = set(used_diffs)
                for v in new_rows[i]:
                    if v != new_val:
                        new_used_diffs.add(abs(new_val - v))
                        
                # Calculate new cost
                new_cost = max(max(r) for r in new_rows)
                
                # Prune if worse than or equal to best known complete solution
                if new_cost < best_cost:
                    next_beam.append((new_cost, new_rows, new_used_diffs))
        
        if not next_beam:
            # Beam empty, search failed to find any extension
            # We might have to break or try to recover
            break
            
        # Sort by cost
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        # Check if any state in beam is a complete solution
        for cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                
    if best_solution is None:
        # Fallback to a simple greedy if beam search failed
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            # Find row with smallest length
            i = min(range(n_rows), key=lambda idx: (len(rows[idx]), rows[idx][-1]))
            if len(rows[i]) >= row_len:
                continue
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - v for v in rows[i]}
                if not (new_diffs & used):
                    rows[i].append(x)
                    used.update(new_diffs)
                    break
                x += 1
        best_solution = rows
        best_cost = max(max(r) for r in rows)

    return best_solution
