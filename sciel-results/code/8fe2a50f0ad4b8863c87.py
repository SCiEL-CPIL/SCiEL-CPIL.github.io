import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _solve_with_backtrack(n_rows, k, max_entry, time_limit=2.0):
    """
    Backtracking search to find a valid construction where all entries are <= max_entry.
    Returns a list of rows if found, else None.
    """
    start_time = time.time()
    
    # used_diffs[d] = True if difference d is used
    # We use a list for speed, size max_entry + 1
    used_diffs = [False] * (max_entry + 1)
    rows = []
    
    # We will build rows one by one.
    # For each row, we build it element by element.
    # row_idx: index of the row being built
    # pos: index of the next element in the current row (1 to k)
    # last_val: the last element added to the current row
    
    # To allow restarting with different randomizations, we can shuffle the candidate order.
    # However, standard backtracking with MRV is deterministic. 
    # We can add a small randomization: pick the next row to complete? No, order matters for pruning.
    # Instead, we can randomize the order of candidates for the next value.
    
    def backtrack(row_idx, pos, last_val, current_row):
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows built successfully
            return True
        
        if pos == k + 1:
            # Current row is complete
            # Note: We don't need to do anything special here, 
            # the diffs were added as we went.
            rows.append(current_row[:])
            return backtrack(row_idx + 1, 1, 0, [0])
        
        # Determine candidates for the next value
        # Must be > last_val and <= max_entry
        # Also, the difference (candidate - last_val) must be valid? 
        # No, ALL differences with all previous elements in current_row must be valid.
        
        candidates = []
        # Optimization: We can iterate from last_val + 1 to max_entry.
        # But checking all diffs is O(k).
        # Let's collect valid candidates.
        
        # Heuristic: Try smaller values first to keep the scope tight?
        # Or randomize? Randomizing helps escape local minima in backtracking if we restart.
        # For a single run, deterministic is often faster if the heuristic is good.
        # Let's try a mix: sorted by value, but with a slight random perturbation if we fail quickly.
        
        for x in range(last_val + 1, max_entry + 1):
            # Check if x is valid with respect to current_row
            valid = True
            for y in current_row:
                d = x - y
                if d > max_entry: # Should not happen if x <= max_entry and y >= 0
                    break
                if used_diffs[d]:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return False
            
        # Shuffle candidates to allow different paths on restart?
        # If we are doing a single deep search, deterministic is better.
        # But if we are doing multiple short searches, shuffling is good.
        random.shuffle(candidates)
        
        for x in candidates:
            # Tentatively add x
            current_row.append(x)
            
            # Mark differences
            diffs_to_mark = []
            for y in current_row[:-1]: # y is in current_row before x
                d = x - y
                diffs_to_mark.append(d)
                used_diffs[d] = True
            
            if backtrack(row_idx, pos + 1, x, current_row):
                return True
            
            # Unmark differences
            for d in diffs_to_mark:
                used_diffs[d] = False
            current_row.pop()
            
        return False

    # Start backtracking
    # We need to pass a mutable structure for rows to be filled
    # The function above appends to 'rows' on success.
    # We need to clear 'rows' before starting a new attempt if we restart.
    
    # Initial call: row_idx=0, pos=1 (next index to fill, since 0 is fixed), last_val=0, current_row=[0]
    if backtrack(0, 1, 0, [0]):
        return rows
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Get a greedy upper bound
    best_rows = _greedy_init(n_rows, k)
    best_val = _global_max(best_rows)
    
    # 2. Try to improve using backtracking with decreasing max_entry
    # We will try M = best_val - 1, best_val - 2, ...
    # For each M, we give it a time slice.
    
    # If best_val is small, we can be more aggressive.
    # If best_val is large, we might need more time per M.
    
    # Strategy: 
    # - Allocate a total search time (e.g., 140s).
    # - Iterate M downwards.
    # - Give each M a time slice that decreases as we go lower? 
    #   Actually, lower M is harder, so maybe give *more* time?
    #   But if we find a solution at M, we stop.
    #   Let's use a fixed time slice per M, e.g., 1.0s, and restart with randomization.
    #   If we run out of time, return best found.
    
    current_m = best_val - 1
    
    # We can also randomize the greedy init to get different upper bounds?
    # Greedy is deterministic.
    
    # Let's try to find a solution for M = current_m.
    # We will use a loop with restarts for the backtracker.
    
    found_improvement = False
    search_deadline = deadline - 5.0 # Keep 5s buffer
    
    while current_m >= 100 and time.time() < search_deadline:
        # Try to solve for current_m
        # We can try multiple times with randomization if we implement it.
        # My current _solve_with_backtrack uses random.shuffle on candidates.
        # So each call is a different random instance.
        
        # Give it a time limit. Let's say 2.0s per attempt.
        # But if we have 140s, we can try ~70 values of M?
        # If best_val is 112, we try 111, 110, ...
        
        # It's better to try the most promising M first.
        # The greedy bound is often close.
        
        rows_found = _solve_with_backtrack(n_rows, k, current_m, time_limit=2.0)
        
        if rows_found:
            # We found a valid solution with max entry <= current_m
            # The actual max entry might be less than current_m, but <= is enough.
            # We can update best_rows and best_val.
            actual_max = _global_max(rows_found)
            if actual_max < best_val:
                best_val = actual_max
                best_rows = rows_found
                # We can try to go even lower?
                # The solution found has max actual_max.
                # We can set current_m = actual_max - 1 to try to improve further.
                current_m = actual_max - 1
                found_improvement = True
            else:
                # Found a solution for M, but it didn't beat the current best?
                # This shouldn't happen if current_m < best_val.
                # Unless the greedy bound was already optimal or lower.
                # If current_m < best_val, then any solution for M has max <= M < best_val.
                # So it must be an improvement.
                pass
        else:
            # No solution found for current_m in the time limit.
            # Decrease M and try again?
            # Wait, if no solution exists for M, there is no solution for M+1 either?
            # NO. If a solution exists for M+1, it might not exist for M.
            # We are looking for the *minimum* M.
            # If we fail to find a solution for M, it might be because:
            # 1. No solution exists for M.
            # 2. Solution exists but backtracker didn't find it in time.
            
            # If we assume the backtracker is complete (given infinite time), 
            # then failing means no solution for M.
            # But we have a time limit.
            
            # Strategy: If we fail for M, we should NOT decrement M.
            # We should try M again with more time or different randomization?
            # Or we can skip M and try M-1? 
            # If no solution for M, no solution for M-1.
            # So if we fail for M, we should probably stop decreasing M?
            # No, that's wrong.
            # If no solution for M, then the minimum M is > M.
            # So we should increase M?
            
            # Let's re-evaluate.
            # We start with M = best_val - 1.
            # If we find a solution, we update best_val to the actual max of that solution.
            # Then we try to find a solution for that new best_val - 1.
            
            # If we FAIL to find a solution for M, it is likely that M is too small.
            # So we should NOT go to M-1. We should probably stop searching for lower M
            # and accept the current best_val.
            # OR, we can try to run the backtracker for longer?
            
            # Given the time constraint, if we fail quickly, we might try again.
            # But if we fail after 2s, it's likely infeasible.
            
            # Let's break if we fail for a few consecutive M values?
            # Actually, if we fail for M, we should assume M is infeasible.
            # Since we are going downwards, if M is infeasible, M-1 is also infeasible.
            # So we should stop.
            
            # However, the backtracker might be incomplete due to time limit.
            # To be safe, we can try a few more times with randomization.
            
            # Let's try 3 times for each M.
            # If all 3 fail, we assume M is infeasible and stop.
            
            # To implement this, we need a counter.
            
            # For simplicity in this loop structure, if we fail, we break.
            # This might be too aggressive if the first attempt was unlucky.
            
            # Let's modify the loop to try multiple attempts for the same M.
            
            # We'll handle this by breaking out and using a more sophisticated outer loop.
            break

    # The simple loop above breaks on first failure.
    # Let's implement a more robust search.
    
    # Reset time check
    if time.time() < search_deadline:
        # Try to improve further from the current best_val
        # We already did this in the loop above if it found improvements.
        # If it broke due to failure, we are stuck.
        
        # Let's try a few random restarts of the backtracker on the current best_val - 1
        # to see if we can find a solution that the first attempt missed.
        
        target_m = best_val - 1
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts and time.time() < search_deadline:
            rows_found = _solve_with_backtrack(n_rows, k, target_m, time_limit=2.0)
            if rows_found:
                actual_max = _global_max(rows_found)
                if actual_max < best_val:
                    best_val = actual_max
                    best_rows = rows_found
                    target_m = best_val - 1
                    attempts = 0 # Reset attempts on success
                else:
                    # Found solution but didn't improve?
                    # Should not happen if target_m < best_val
                    break
            else:
                attempts += 1
                # If we failed, try a slightly higher M? No, we want lower.
                # If we failed for target_m, it's likely infeasible.
                # But maybe we just need more time?
                # We can't allocate more time without hurting other parts.
                # Let's break if we fail a few times.
                if attempts >= max_attempts:
                    break

    return best_rows

# EVOLVE-BLOCK-END
