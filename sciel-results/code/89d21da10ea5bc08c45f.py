import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_rows = None
    best_max_val = float('inf')
    
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Vary the randomization window to explore different spaces
        # Smaller window is more greedy, larger is more random
        window = 3 + (attempts % 7) 
        
        rows = _greedy_randomized(n_rows, k, rng, window)
        
        if rows is not None:
            # Apply local search to tighten the solution
            improved_rows, improved_max = _local_search(rows, n_rows, k, rng, deadline)
            
            if improved_max < best_max_val:
                best_max_val = improved_max
                best_rows = improved_rows
                
            # Early termination if we hit a very good bound
            # For 7x6, 105 is LB. 110 is very good.
            if best_max_val <= 110:
                break

    if best_rows is None:
        best_rows = _greedy_deterministic(n_rows, k)
        
    return best_rows

def _greedy_randomized(n_rows, k, rng, window):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        success = True
        
        while len(row) < k + 1:
            valid_candidates = []
            x = current_max + 1
            
            while len(valid_candidates) < window:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    valid_candidates.append(x)
                x += 1
                
            if not valid_candidates:
                success = False
                break
            
            chosen = rng.choice(valid_candidates)
            row.append(chosen)
            current_max = chosen
            # Update used set
            for a in row[:-1]:
                used.add(chosen - a)
            
        if not success:
            return None
        rows.append(row)
        
    return rows

def _greedy_deterministic(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, rng, deadline):
    # Deep copy
    current_rows = [list(r) for r in rows]
    
    # Build initial used set
    used = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i):
                used.add(r[i] - r[j])
                
    current_max = max(max(r) for r in current_rows)
    
    # Iterative improvement
    # We will try to reduce the maximum element in the set.
    # We focus on the last element of rows, as they are the most flexible.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find the row with the largest last element
        max_val = -1
        max_row_idx = -1
        for i, r in enumerate(current_rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_row_idx = i
                
        r = current_rows[max_row_idx]
        j = len(r) - 1 # Last element index
        
        # Try to reduce r[j]
        # Scan downwards from max_val - 1
        for y in range(max_val - 1, r[j-1], -1):
            # Check if y is valid
            # Old diffs involving j: {r[j] - r[i] for i in range(j)}
            old_diffs = {r[j] - r[i] for i in range(j)}
            
            # New diffs involving j: {y - r[i] for i in range(j)}
            new_diffs = {y - r[i] for i in range(j)}
            
            # Check for conflicts with other parts of the system
            # A conflict exists if any new_diff is in `used` but NOT in `old_diffs`
            # (Because old_diffs are being removed)
            
            # Note: new_diffs might contain values that are in old_diffs. That's fine.
            # We check: (new_diffs & used) - old_diffs
            # If this set is empty, it's valid.
            
            conflicts = (new_diffs & used) - old_diffs
            if not conflicts:
                # Accept the change
                # Update used set
                used -= old_diffs
                used |= new_diffs
                
                r[j] = y
                current_max = max(max(row) for row in current_rows)
                improved = True
                break # Move to next iteration to find new max
                
    return current_rows, current_max

# EVOLVE-BLOCK-END
