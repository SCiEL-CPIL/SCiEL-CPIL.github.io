import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # 1. Generate initial solution
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Precompute diffs for current solution
    all_diffs = set()
    for r in best_solution:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.add(r[j] - r[i])
                
    current_solution = [r[:] for r in best_solution]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Choose a move
        move_type = random.random()
        candidate = [r[:] for r in current_solution]
        candidate_diffs = set(all_diffs) # We will update this
        
        # We need to be careful with diff management.
        # It's safer to just recompute diffs for the candidate if we make a move, 
        # or maintain a delta.
        # For simplicity and correctness, let's use a helper that takes the solution 
        # and returns the score and diffs, but that's O(N*K^2) per move which might be slow.
        # Better: Maintain the set of diffs and update it incrementally.
        
        if move_type < 0.7:
            # Row Re-solve
            r_idx = random.randint(0, n_rows - 1)
            
            # Remove old row diffs
            old_row = current_solution[r_idx]
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    d = old_row[j] - old_row[i]
                    candidate_diffs.remove(d)
                    
            # Try to construct a new row
            # We want to try to keep the max low.
            # We can try a few random restarts of the row construction.
            best_new_row = None
            best_new_row_max = float('inf')
            
            # Limit the search space for the new row to not be too large
            # We can try to fit it within the current max or slightly above
            max_allowed = current_score + 10 
            
            for _ in range(20): # 20 attempts to rebuild this row
                new_row = _random_greedy_row(k, candidate_diffs, max_allowed)
                if new_row is not None:
                    m = max(new_row)
                    if m < best_new_row_max:
                        best_new_row_max = m
                        best_new_row = new_row
                        if m < old_row[-1]:
                            break # Found a strictly better row, stop trying
                
            if best_new_row is not None:
                candidate[r_idx] = best_new_row
                # Add new row diffs
                for i in range(len(best_new_row)):
                    for j in range(i + 1, len(best_new_row)):
                        candidate_diffs.add(best_new_row[j] - best_new_row[i])
                        
                new_score = max(max(r) for r in candidate)
                
                # Acceptance
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = candidate
                    all_diffs = candidate_diffs
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_solution = [r[:] for r in current_solution]
            else:
                # Failed to find a valid row, revert (do nothing)
                pass
        else:
            # Single Position Move
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = current_solution[r_idx][pos]
            
            # Remove diffs involving old_val
            row = current_solution[r_idx]
            to_remove = set()
            for i in range(len(row)):
                if i == pos: continue
                if i < pos:
                    to_remove.add(old_val - row[i])
                else:
                    to_remove.add(row[i] - old_val)
            for d in to_remove:
                candidate_diffs.remove(d)
                
            # Find a new valid value
            # Try a random value in a reasonable range
            lower = row[pos-1] + 1
            upper = row[pos+1] - 1 if pos < k else current_score + 5
            if upper < lower:
                upper = current_score + 5
                
            new_val = None
            # Try a few random candidates
            for _ in range(20):
                val = random.randint(lower, upper)
                if val == old_val: continue
                
                # Check validity
                valid = True
                new_diffs = set()
                for i in range(len(row)):
                    if i == pos: continue
                    if i < pos:
                        d = val - row[i]
                    else:
                        d = row[i] - val
                    if d in candidate_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    
                if valid:
                    new_val = val
                    break
                    
            if new_val is not None:
                candidate[r_idx][pos] = new_val
                for d in new_diffs:
                    candidate_diffs.add(d)
                    
                new_score = max(max(r) for r in candidate)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = candidate
                    all_diffs = candidate_diffs
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_solution = [r[:] for r in current_solution]
            
        T *= cooling_rate
        steps += 1
        
    return best_solution

def _random_greedy_row(k, used_diffs, max_limit):
    """
    Attempts to construct a row of k+1 integers starting at 0.
    Uses a randomized greedy approach.
    """
    row = [0]
    current_max = 0
    for _ in range(k):
        found = False
        # Try values from current_max + 1 up to max_limit
        # To add randomness, we can shuffle a small window or pick random offsets
        # But we must ensure increasing order.
        # Let's try a few random candidates near the bottom, then fill up.
        
        # Strategy: Try smallest valid first, but with some randomness to escape traps.
        # We'll just iterate from current_max + 1 to max_limit.
        # If k is small (6), this is fast.
        for x in range(current_max + 1, max_limit + 1):
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                used_diffs.update(new_diffs)
                current_max = x
                found = True
                break
                
        if not found:
            return None
            
    return row

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Limit search to avoid infinite loops if impossible, though greedy should work for small n
            for x in range(current_max + 1, current_max + 500):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
