"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses Randomized Greedy Initialization followed by Local Search (Hill Climbing).
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    best_solution = None
    best_max_val = float('inf')
    
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # --- Phase 1: Randomized Greedy Initialization ---
        random.seed(restart * 7919 + 13) # Distinct prime for better distribution
        
        used_diffs = set()
        rows = []
        valid_init = True
        
        for r in range(n_rows):
            row = [0]
            # We want to construct a row of length k+1
            # Greedily pick the next element
            for pos in range(k):
                # Find candidates that don't conflict
                # Start from last element + 1
                start_val = row[-1] + 1
                found = False
                
                # Try a window of values
                # To introduce randomness, we can shuffle a range of candidates
                # but we must ensure we find a valid one.
                # Let's try values in a random order within a reasonable bound.
                # Bound: maybe start_val to start_val + 50
                candidates = list(range(start_val, start_val + 100))
                random.shuffle(candidates)
                
                for x in candidates:
                    # Check differences with existing elements in this row
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if not conflict:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    # Extend search if needed, or fail
                    # If we fail, this restart is invalid
                    valid_init = False
                    break
            
            if not valid_init:
                break
            
            rows.append(row)
        
        if not valid_init or len(rows) != n_rows:
            continue
        
        # Verify initial solution
        all_diffs = set()
        max_val = 0
        for row in rows:
            if row[0] != 0:
                valid_init = False
                break
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid_init = False
                        break
                    all_diffs.add(d)
                max_val = max(max_val, row[-1])
            if not valid_init:
                break
        
        if not valid_init:
            continue
        
        # --- Phase 2: Local Search Optimization ---
        # We have a valid solution. Try to reduce the max value.
        # Strategy: Try to decrease any element to a smaller valid value.
        
        current_max = max_val
        # We will maintain the global difference set to speed up checks
        # But since we modify one row at a time, we can just recompute or update.
        # Given small size, recomputing or careful updating is fine.
        # Let's use a helper to check validity of a specific row change.
        
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation (though we only accept improvements)
        max_iter = 1000
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try to improve any element in any row
            # Randomize the order of checks to avoid bias
            indices = [(r, c) for r in range(n_rows) for c in range(k, 0, -1)]
            random.shuffle(indices)
            
            for r_idx, c_idx in indices:
                row = rows[r_idx]
                current_val = row[c_idx]
                
                # Lower bound for this position: row[c_idx - 1] + 1
                lower_bound = row[c_idx - 1] + 1
                
                # Try to reduce current_val
                # We can try values from current_val - 1 down to lower_bound
                # To be efficient, we can check if there's *any* valid value lower.
                
                # Optimization: Only try if current_val > lower_bound
                if current_val <= lower_bound:
                    continue
                
                # Try candidates in descending order (closest to current first? or random?)
                # Descending is good for quick wins.
                candidates = range(current_val - 1, lower_bound - 1, -1)
                
                for new_val in candidates:
                    # Check if changing row[r_idx][c_idx] to new_val keeps the system valid
                    
                    # 1. Row internal validity (strictly increasing) is guaranteed by new_val > lower_bound
                    #    and new_val < row[c_idx+1] if c_idx < k.
                    #    Wait, we are only changing c_idx. We must ensure new_val < row[c_idx+1] if it exists.
                    if c_idx < k and new_val >= row[c_idx + 1]:
                        continue
                    
                    # 2. Global difference validity
                    # Remove old differences involving row[r_idx][c_idx]
                    # Add new differences involving new_val
                    
                    # For efficiency, we can just rebuild the difference set for this row
                    # and check against the global set minus this row's old diffs.
                    
                    # Let's just do a full check for simplicity, as n_rows and k are small.
                    # But we can be smarter:
                    # old_diffs_in_row = {row[j] - row[i] for i<j in row}
                    # new_diffs_in_row = ...
                    
                    # Let's implement a fast check:
                    # 1. Create a set of all differences from all rows EXCEPT the one being modified.
                    # This is expensive to recompute every time.
                    
                    # Alternative: Maintain a global `used_diffs` set.
                    # When we propose a change:
                    # 1. Calculate `old_row_diffs` for the current row.
                    # 2. Calculate `new_row_diffs` for the modified row.
                    # 3. Check if `new_row_diffs` intersects with (`used_diffs` - `old_row_diffs`).
                    
                    # We need to maintain `used_diffs` globally.
                    # Let's do that.
                    
                    # Calculate old diffs for this row
                    old_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_diffs.add(row[j] - row[i])
                    
                    # Calculate new diffs for this row
                    new_row = row[:]
                    new_row[c_idx] = new_val
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    
                    # Check conflicts
                    # The global set `used_diffs` contains diffs from all rows.
                    # The "other" diffs are `used_diffs` - `old_diffs`.
                    # We need `new_diffs` to be disjoint from `other_diffs`.
                    # And `new_diffs` must be disjoint from itself (always true for a set).
                    
                    other_diffs = used_diffs - old_diffs
                    if not (new_diffs & other_diffs):
                        # Valid move!
                        # Update global state
                        used_diffs = (used_diffs - old_diffs) | new_diffs
                        rows[r_idx] = new_row
                        current_max = max(current_max, new_row[-1]) # Actually we should recompute max or track it
                        # Since we are decreasing a value, the max of this row might decrease.
                        # The global max is max of all row maxes.
                        # Let's just recompute global max after a change to be safe, or track it.
                        # Recomputing is O(n_rows * k), which is small.
                        current_max = 0
                        for r in rows:
                            current_max = max(current_max, r[-1])
                        
                        improved = True
                        break # Break out of candidate loop, restart index loop
                
                if improved:
                    break # Break out of index loop
        
        # End Local Search
        
        if current_max < best_max_val:
            best_max_val = current_max
            best_solution = [list(r) for r in rows]
            
        # Early exit if we hit the theoretical lower bound?
        # Lower bound: total_diffs / n_rows
        total_diffs = n_rows * k * (k + 1) // 2
        lb = (total_diffs + n_rows - 1) // n_rows
        if best_max_val <= lb:
            break
            
    if best_solution is not None:
        return best_solution
    
    # Fallback: Return a valid solution if we found one, else greedy
    # The loop above should always find a valid solution if one exists.
    # If not, return a simple greedy one (which might have high max but is valid).
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
