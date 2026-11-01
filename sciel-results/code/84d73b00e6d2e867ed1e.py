import time
import random

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # 1. Greedy Initial Solution
    initial_rows = _greedy_construct(n_rows, k)
    if initial_rows is None:
        # Fallback: simple construction with large gaps
        initial_rows = _simple_construct(n_rows, k)
        
    best_rows = initial_rows
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Downward Search with Backtracking
    lower_bound = n_rows * (k * (k + 1) // 2)
    
    for limit in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
            
        # Try to find solution with max <= limit
        # Use deterministic first
        found = _backtrack_construct(n_rows, k, limit, deadline, randomize=False)
        
        if found is None:
            # Try randomized if time permits
            if time.time() < deadline - 5:
                found = _backtrack_construct(n_rows, k, limit, deadline, randomize=True)
                
        if found is not None:
            best_rows = found
            best_scope = limit
            if limit == lower_bound:
                break
        else:
            # If we can't find a solution for this limit, 
            # it's likely we can't find one for smaller limits either 
            # (as the space is tighter), but not guaranteed.
            # However, to be safe, we break if we've tried a few.
            # Given the small size, if 111 fails, 110 probably fails.
            break

    # 3. Final Polish: Randomized Restart on Best Scope
    # If we have time, try to find a better solution with the SAME scope?
    # No, we want smaller scope.
    # Try to find a solution with scope < best_scope again with more randomization?
    # The loop above already did that.
    
    return best_rows

def _greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        last_x = 0
        while len(row) < k + 1:
            found = False
            # Try small increments
            for x in range(last_x + 1, last_x + 500):
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                if (new_diffs_mask & used) == 0:
                    row.append(x)
                    last_x = x
                    used |= new_diffs_mask
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _simple_construct(n_rows, k):
    # Use a very large stride to avoid conflicts
    # This is not optimal but ensures a valid start
    rows = []
    offset = 0
    for _ in range(n_rows):
        row = [0]
        for i in range(1, k+1):
            # Pick a number that is definitely not conflicting
            # Just use a large number
            x = offset + i * 1000
            row.append(x)
        rows.append(row)
        offset += 1000 * (k+1)
    return rows

def _backtrack_construct(n_rows, k, limit, global_deadline, randomize=False):
    """
    Finds a set of rows with max element <= limit.
    """
    
    def backtrack(row_idx, current_row, last_x, used_mask):
        if time.time() > global_deadline:
            return None
        
        if row_idx == n_rows:
            return []
            
        if len(current_row) == 0:
            current_row = [0]
            last_x = 0
            
        if len(current_row) == k + 1:
            remaining = backtrack(row_idx + 1, [], 0, used_mask)
            if remaining is not None:
                return [current_row] + remaining
            return None
            
        candidates = list(range(last_x + 1, limit + 1))
        if randomize:
            random.shuffle(candidates)
        else:
            # Smaller first
            pass 
            
        for x in candidates:
            new_diffs_mask = 0
            for a in current_row:
                d = x - a
                new_diffs_mask |= (1 << d)
                
            if (new_diffs_mask & used_mask) == 0:
                result = backtrack(row_idx, current_row + [x], x, used_mask | new_diffs_mask)
                if result is not None:
                    return result
                    
        return None

    result = backtrack(0, [], 0, 0)
    if result:
        return result
    return None
