"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

def _greedy_row(used, k):
    """Build one row greedily given the set of used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _backtrack_row(used, k, max_val, seed):
    """
    Try to build a row of length k+1 starting at 0, with max value <= max_val.
    Returns the row if found, else None.
    Uses backtracking with randomization.
    """
    rand = random.Random(seed)
    row = [0]
    
    def backtrack(last_val):
        if len(row) == k + 1:
            return True
        
        # Generate candidates
        candidates = list(range(last_val + 1, max_val + 1))
        
        # Heuristic: Sort by value, but introduce randomization to escape local minima.
        # We can shuffle a subset or just shuffle entirely.
        # Shuffling entirely might be too random.
        # Let's try: 50% chance to sort, 50% chance to shuffle.
        # Or better: Sort, but swap the first few elements randomly.
        # For now, let's just shuffle to ensure we explore different paths.
        # Given the small size, this should be okay.
        if rand.random() < 0.5:
            rand.shuffle(candidates)
        
        for x in candidates:
            new_diffs = [x - a for a in row]
            
            conflict = False
            for d in new_diffs:
                if d in used:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add to used
            for d in new_diffs:
                used.add(d)
            row.append(x)
            
            if backtrack(x):
                return True
            
            # Undo
            row.pop()
            for d in new_diffs:
                used.remove(d)
        
        return False
    
    if backtrack(0):
        return row
    return None

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # Baseline using greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
        # Update used for the next row in greedy baseline
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    best_rows = rows
    best_max = max(max(r) for r in rows)
    
    # Backtracking search with decreasing max_val bound
    # We know the counting bound is 105 for k=5, n_rows=7.
    # So we search from best_max - 1 down to 105.
    
    for bound in range(best_max - 1, 104, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try multiple restarts for this bound
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            used = set()
            rows = []
            success = True
            
            for i in range(n_rows):
                row = _backtrack_row(used, k, bound, attempt * 100 + i)
                if row is None:
                    success = False
                    break
                rows.append(row)
            
            if success:
                current_max = max(max(r) for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows
                    found = True
                    break
        
        # If we didn't find a solution for this bound, lower bounds won't work either
        # (assuming the backtracker is complete enough, which it is for small k)
        if not found:
            break

    return best_rows
# EVOLVE-BLOCK-END
