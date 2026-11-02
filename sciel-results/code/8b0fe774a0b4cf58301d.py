import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization.
    Returns a list of rows, where each row is a list of k+1 integers.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        current_max = 0
        # We need k more elements
        while len(row) < k + 1:
            # Try increasing values from current_max + 1
            found = False
            # We can limit the search range slightly for speed, but greedy needs to be thorough
            # A safe upper bound for the next element in a greedy construction
            # is roughly current_max + k + 1 + len(used_diffs) but that's huge.
            # Just iterate upwards.
            for x in range(current_max + 1, current_max + 500): 
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
                    for d in new_diffs:
                        used_diffs.add(d)
                    current_max = x
                    found = True
                    break
            
            if not found:
                # Fallback: if we can't find a small x, jump higher
                # This shouldn't happen often in greedy init if we start fresh
                x = current_max + 500
                while True:
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
                        for d in new_diffs:
                            used_diffs.add(d)
                        current_max = x
                        break
                    x += 1
        rows.append(row)
        
    return rows

def _compute_row_diffs(row):
    """Compute the set of differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _reconstruct_row(k, forbidden_diffs, target_max=None):
    """
    Try to construct a row of length k+1 starting at 0.
    The differences must not be in forbidden_diffs.
    If target_max is provided, the row must be <= target_max.
    Returns the row or None if not found.
    """
    row = [0]
    # We want to minimize the last element.
    # Greedy strategy: pick the smallest possible next element.
    
    # To make it slightly stochastic for ILS, we can randomize the choice among the first few valid options
    # But for local search, deterministic greedy is often best for "repairing".
    # Let's use a greedy approach that looks for the smallest valid x.
    
    for _ in range(k):
        # The next element must be > row[-1]
        # And if target_max is set, <= target_max
        start = row[-1] + 1
        end = target_max if target_max is not None else float('inf')
        
        found_val = None
        # Search range: usually small increments are sufficient
        # If target_max is tight, we search up to target_max.
        # If not, we search a bit ahead.
        search_limit = end if end < float('inf') else row[-1] + 100
        
        for x in range(start, search_limit + 1):
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in forbidden_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                found_val = x
                # Store diffs to add later? No, we just return the row.
                # The caller will add the diffs.
                break
        
        if found_val is None:
            return None
        row.append(found_val)
        
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization
    best_rows = _greedy_init(n_rows, k)
    best_val = max(max(r) for r in best_rows)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Pre-compute diffs for each row
    current_row_diffs = [_compute_row_diffs(r) for r in current_rows]
    
    # Global used diffs
    all_used = set()
    for rd in current_row_diffs:
        all_used |= rd
        
    # 2. Iterated Local Search
    # Strategy:
    # - Pick a row to "break".
    # - Remove its diffs from all_used.
    # - Try to reconstruct it with a lower max value.
    # - If successful and improves global max, accept.
    # - If not, try to reconstruct it with the same or slightly higher max to "shuffle" it (perturbation).
    
    T = 50.0
    T_min = 1.0
    cool = 0.999
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        
        # Pick a row to modify
        ri = random.randrange(n_rows)
        
        # Remove row ri from global constraints
        old_diffs = current_row_diffs[ri]
        all_used -= old_diffs
        
        # Determine target max for reconstruction
        # We want to try to reduce the max of this row, or the global max.
        # Let's try to reconstruct with a target max slightly lower than the current row's max.
        # If that fails, try the current max.
        
        old_row_max = max(current_rows[ri])
        
        # Try reducing the max of this row
        # We can try a range of target maxes
        # If this row is the one determining the global max, reducing it is crucial.
        
        new_row = None
        # Try to find a row with max < old_row_max
        # Start from old_row_max - 1 down to some limit
        # Or just try old_row_max - 1, old_row_max - 2, etc.
        
        # To save time, let's try a few random target maxes below the current one
        # And one at the current one (for shuffling)
        
        # 1. Try to improve
        improved = False
        for _ in range(10): # Try 10 times with decreasing target max
            target_max = old_row_max - random.randint(1, 5)
            if target_max < 0:
                break
            new_row = _reconstruct_row(k, all_used, target_max)
            if new_row is not None:
                # Check if this new row actually has a smaller max
                # (It should, by construction of target_max)
                # But we need to ensure it doesn't conflict with itself (handled by _reconstruct_row)
                # And it doesn't conflict with all_used (handled by _reconstruct_row)
                
                # Calculate new global max
                new_row_max = max(new_row)
                # The global max is max of all other rows and this new row
                # We can approximate: if this row was the unique max, and new_row_max < old_row_max, 
                # we might reduce global max.
                
                # Let's just accept if it's valid and has a lower max than the old row
                # This is a local improvement.
                
                # Update state
                new_diffs = _compute_row_diffs(new_row)
                # Check for conflicts with all_used (should be none by construction)
                # But _reconstruct_row uses the current all_used which excludes old_diffs.
                # So new_diffs should be disjoint from all_used.
                
                # However, we must ensure new_diffs are disjoint from each other (handled)
                # and from all_used (handled).
                
                # Accept
                current_rows[ri] = new_row
                current_row_diffs[ri] = new_diffs
                all_used |= new_diffs
                
                # Update global max
                # We can just recompute or track. Recomputing is O(n_rows * k)
                new_global_max = max(max(r) for r in current_rows)
                
                if new_global_max < current_val:
                    current_val = new_global_max
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                improved = True
                break
        
        # 2. If not improved, try to shuffle (perturbation)
        if not improved:
            # Try to reconstruct with the same max, but different structure
            # This helps escape local minima
            new_row = _reconstruct_row(k, all_used, old_row_max)
            if new_row is not None:
                # Accept if it's different? Or just accept to explore.
                # Let's accept with some probability or always accept if valid.
                # Since we are in ILS, we accept valid moves.
                new_diffs = _compute_row_diffs(new_row)
                current_rows[ri] = new_row
                current_row_diffs[ri] = new_diffs
                all_used |= new_diffs
                # Global max likely unchanged
                no_improve += 1
            else:
                # Failed to reconstruct, restore old row
                all_used |= old_diffs
                no_improve += 1
        
        T = max(T_min, T * cool)
        
        # Restart if stuck
        if no_improve > 2000:
            # Reset to best
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            current_row_diffs = [_compute_row_diffs(r) for r in current_rows]
            all_used = set()
            for rd in current_row_diffs:
                all_used |= rd
            no_improve = 0
            T = 50.0

    return best_rows

# EVOLVE-BLOCK-END
