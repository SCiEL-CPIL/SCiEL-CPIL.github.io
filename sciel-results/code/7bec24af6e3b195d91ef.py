import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Helper to check if a row is valid given the set of used differences
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to find a valid row given used diffs and a max value constraint
    def find_row(used, k, max_val, rng):
        # Greedy construction with some randomization
        # We want a row of length k+1 starting at 0, strictly increasing, max <= max_val
        # such that all differences are not in 'used'
        
        # Backtracking search might be too slow if max_val is tight.
        # Let's try a randomized greedy approach first.
        for _ in range(100): # Try a few times
            row = [0]
            current = 0
            success = True
            for _ in range(k):
                # Try to pick next number
                found = False
                # Randomize the order of candidates? 
                # Or just greedy smallest first?
                # Greedy smallest first is good for keeping scope small.
                # But we might get stuck.
                # Let's try a few candidates around the minimum possible.
                min_next = row[-1] + 1
                # We need to ensure we can still fit k - len(row) numbers after this.
                # Max possible value for this position is max_val - (k - len(row))
                
                # Generate candidates
                candidates = []
                # Start from min_next up to some limit
                # To keep it fast, we don't scan all the way to max_val immediately
                # We scan a window
                limit = min(max_val - (k - len(row)), min_next + 50)
                
                for x in range(min_next, limit + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if len(candidates) > 10: break
                
                if not candidates:
                    success = False
                    break
                
                # Pick one
                # Greedy: smallest
                # Random: weighted?
                x = candidates[0] # Greedy is usually better for scope minimization
                row.append(x)
                used.update(new_diffs)
            
            if success and row[-1] <= max_val:
                return row
        
        return None

    # Initial construction
    # We want to balance the scopes. 
    # Simple strategy: Build rows one by one, but try to keep the max value low.
    # Or: Use a global 'used' set.
    
    used = set()
    rows = []
    
    # Estimate a reasonable initial scope.
    # Counting bound: total diffs = n_rows * k(k+1)/2
    # Max scope m must satisfy m(m+1)/2 >= total_diffs
    total_diffs = n_rows * k * (k + 1) // 2
    # Solve m(m+1)/2 >= total_diffs
    # m ~ sqrt(2 * total_diffs)
    import math
    est_scope = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    # Add some buffer
    current_max_scope = est_scope + 10
    
    for i in range(n_rows):
        # Try to find a row with scope <= current_max_scope
        # If not, increase scope
        row = None
        while row is None:
            row = find_row(used, k, current_max_scope, random)
            if row is None:
                current_max_scope += 5
                if current_max_scope > 1000: # Fallback
                     # Just greedy without bound
                     row = [0]
                     x = 0
                     while len(row) < k + 1:
                         x += 1
                         new = {x - a for a in row}
                         if not new & used:
                             row.append(x)
                             used |= new
                     break
        
        rows.append(row)
        used.update(row_diffs(row))
        
        if time.time() > deadline - 50:
            break

    # Ensure we have n_rows rows
    while len(rows) < n_rows:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    # Local Search Phase
    # Goal: Reduce the maximum entry.
    # Strategy: Identify the row with the largest max entry.
    # Try to rebuild that row with a smaller max entry.
    
    best_rows = rows
    best_max_val = max(max(r) for r in rows)
    
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    while time.time() < deadline - 5:
        # Find the row with the largest max value
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        current_row_max = rows[max_idx][-1]
        
        if current_row_max <= 1: break
        
        # Try to rebuild this row with a tighter bound
        # Remove its diffs from 'used'
        old_diffs = row_diffs(rows[max_idx])
        new_used = used - old_diffs
        
        # Try bounds from current_row_max - 1 down to some limit
        found = False
        for target_max in range(current_row_max - 1, max(0, current_row_max - 20), -1):
            new_row = find_row(new_used, k, target_max, random)
            if new_row is not None:
                # Check if this new row actually improves the global max
                # It must be <= target_max, which is < current_row_max
                # So it improves the local max, and potentially the global max
                rows[max_idx] = new_row
                used = new_used | row_diffs(new_row)
                found = True
                break
        
        if not found:
            # If we couldn't improve this row, maybe try another row?
            # Or try a different randomization?
            # For now, if we can't improve the worst row, we might be stuck.
            # We could try to swap rows or something more complex, but let's keep it simple.
            # If we fail to improve the max row, we might just restart the local search
            # or accept that we've reached a local minimum for this strategy.
            # To avoid infinite loop on failure, we can try to slightly perturb the problem?
            # Or just break if we've tried a few times without success?
            
            # Let's try to find ANY row that can be improved, not just the worst.
            improved_any = False
            for i in range(len(rows)):
                if i == max_idx: continue
                row_max = rows[i][-1]
                if row_max <= 1: continue
                
                old_diffs_i = row_diffs(rows[i])
                new_used_i = used - old_diffs_i
                
                # Try to improve this row
                for target_max in range(row_max - 1, max(0, row_max - 10), -1):
                    new_row_i = find_row(new_used_i, k, target_max, random)
                    if new_row_i is not None:
                        rows[i] = new_row_i
                        used = new_used_i | row_diffs(new_row_i)
                        improved_any = True
                        break
                if improved_any:
                    break
            
            if not improved_any:
                # Stuck. Try a random restart of a row?
                # Pick a random row and try to rebuild it with a slightly larger scope 
                # to see if it opens up space for others? No, we want to minimize.
                # Maybe just break if we've been stuck for a while?
                # Let's just break to be safe.
                break
        
        # Update best
        current_max = get_max_val(rows)
        if current_max < best_max_val:
            best_max_val = current_max
            best_rows = [r[:] for r in rows]
            
        # If we haven't improved in a while, we might want to break
        # But the loop condition is time-based.
        
    # Return the best found
    return best_rows

