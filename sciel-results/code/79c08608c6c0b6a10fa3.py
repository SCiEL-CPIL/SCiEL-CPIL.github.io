import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs_mask(row):
    """Return bitmask of all pairwise differences in a row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = 0
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        mask = _row_diffs_mask(row)
        if (used & mask) != 0:
            return False
        used |= mask
    return True

def _max_entry(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _build_greedy_mask(n_rows, k, max_val=None, rng=None):
    """Fast greedy using bitmasks. Returns (rows, used_mask, max_val) or None."""
    if rng is None:
        rng = random.Random(42)
        
    used_mask = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        found = False
        # We need to fill k more elements
        while len(row) < k + 1:
            x += 1
            if max_val is not None and x > max_val:
                break
            # Check collisions
            new_diffs_mask = 0
            for a in row:
                new_diffs_mask |= (1 << (x - a))
            
            if (new_diffs_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_diffs_mask
                found = True
            else:
                # If we hit max_val and didn't fill, fail
                if max_val is not None and x == max_val:
                    break
        
        if not found or len(row) != k + 1:
            return None, None, None
        rows.append(row)
        
    return rows, used_mask, max(max(r) for r in rows)

def _try_rebuild_row(rows, ri, k, target_max, used_excl_mask, rng):
    """
    Try to rebuild row ri with max <= target_max.
    Returns new_row or None.
    """
    # We use a randomized greedy approach for the single row
    # To increase success rate, we shuffle the order of potential "gaps" or just restart
    for _ in range(5): # 5 attempts
        row = [0]
        x = 0
        success = True
        while len(row) < k + 1:
            x += 1
            if x > target_max:
                success = False
                break
            
            new_diffs_mask = 0
            for a in row:
                new_diffs_mask |= (1 << (x - a))
            
            if (new_diffs_mask & used_excl_mask) == 0:
                row.append(x)
                # Note: We don't update used_excl_mask here because we are just checking
                # if this specific row is valid against the *other* rows.
                # However, we must ensure the row itself is Sidon.
                # The check (new_diffs_mask & used_excl_mask) == 0 ensures no collision with others.
                # But we also need to ensure no collision *within* the row being built.
                # The greedy construction naturally ensures the row is Sidon because we only add x
                # if it doesn't collide with *previous* elements of the row?
                # Wait, `used_excl_mask` does NOT contain diffs from `row` itself.
                # So `new_diffs_mask` contains diffs between x and all a in row.
                # We need to check if these new diffs collide with *each other*? No, they are all distinct by construction (x-a is unique for each a).
                # We need to check if they collide with *previous* diffs in the row?
                # The previous diffs are not in `used_excl_mask`.
                # So we MUST check if `new_diffs_mask` intersects with diffs already in `row`.
                
                # Let's optimize: keep a `row_mask`
                pass
        
        # The simple greedy above is flawed because it doesn't track internal row collisions properly 
        # if we don't update a local mask. Let's rewrite this helper to be correct.
        pass
        
    return None

def _try_rebuild_row_correct(rows, ri, k, target_max, used_excl_mask, rng):
    """
    Correctly try to rebuild row ri with max <= target_max.
    used_excl_mask contains diffs from all OTHER rows.
    """
    for attempt in range(10):
        row = [0]
        row_mask = 0 # Diffs within the current row
        x = 0
        success = True
        
        while len(row) < k + 1:
            x += 1
            if x > target_max:
                success = False
                break
            
            new_diffs_mask = 0
            for a in row:
                new_diffs_mask |= (1 << (x - a))
            
            # Check collision with other rows
            if (new_diffs_mask & used_excl_mask) != 0:
                continue
            
            # Check collision with current row's internal diffs
            if (new_diffs_mask & row_mask) != 0:
                continue
            
            # Valid move
            row.append(x)
            row_mask |= new_diffs_mask
        
        if success and len(row) == k + 1:
            return row
            
        # Random restart logic: we could randomize the starting points or choices, 
        # but for a single row, the greedy "smallest valid" is often optimal for minimizing max.
        # To diversify, we can skip some small x's randomly.
        
    return None

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    rng = random.Random(42)
    
    # 1. Initial Greedy Solution
    rows, used_mask, cur_max = _build_greedy_mask(n_rows, k, None, rng)
    if rows is None:
        # Fallback to standard greedy if bitmask greedy fails (unlikely)
        return _build_greedy(n_rows, k)
        
    best_rows = [r[:] for r in rows]
    best_max = cur_max
    
    # Precompute row masks for fast exclusion
    row_masks = [_row_diffs_mask(r) for r in rows]
    
    # 2. Local Search / Iterated Improvement
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Try to reduce cur_max by 1
        target_max = cur_max - 1
        
        if target_max < n_rows * k * (k+1) // 2: # Counting bound
             break
             
        improved = False
        for ri in range(n_rows):
            if time.time() > deadline:
                break
                
            # Calculate used mask excluding row ri
            used_excl = used_mask & ~row_masks[ri]
            
            # Try to rebuild row ri with smaller max
            new_row = _try_rebuild_row_correct(rows, ri, k, target_max, used_excl, rng)
            
            if new_row is not None:
                # Update state
                new_row_mask = _row_diffs_mask(new_row)
                
                # Verify global validity (should be true by construction)
                # used_mask = used_excl | new_row_mask
                # Check if new_row_mask intersects used_excl (should be 0)
                if (new_row_mask & used_excl) != 0:
                    continue # Safety check
                
                rows[ri] = new_row
                row_masks[ri] = new_row_mask
                used_mask = used_excl | new_row_mask
                cur_max = max(max(r) for r in rows)
                
                # If cur_max dropped below target_max, we might have improved more
                # But we loop again anyway.
                improved = True
                no_improve_count = 0
                break # Restart loop with new state
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perform a "Kick" / Perturbation
                # Randomly change one row to a random valid row (within current max)
                # This helps escape local optima where greedy rebuild fails.
                if time.time() > deadline:
                    break
                    
                ri = rng.randrange(n_rows)
                used_excl = used_mask & ~row_masks[ri]
                
                # Try to find ANY valid row for this slot, preferably random-ish
                # We can use a randomized greedy
                found_kick = False
                for _ in range(20):
                    new_row = [0]
                    row_mask = 0
                    x = 0
                    success = True
                    # Randomize by skipping some values
                    skip_prob = 0.2
                    
                    while len(new_row) < k + 1:
                        x += 1
                        if x > cur_max:
                            success = False
                            break
                            
                        if rng.random() < skip_prob:
                            continue
                            
                        new_diffs_mask = 0
                        for a in new_row:
                            new_diffs_mask |= (1 << (x - a))
                            
                        if (new_diffs_mask & used_excl) != 0:
                            continue
                        if (new_diffs_mask & row_mask) != 0:
                            continue
                            
                        new_row.append(x)
                        row_mask |= new_diffs_mask
                        
                    if success and len(new_row) == k + 1:
                        # Accept kick even if it doesn't improve max immediately
                        # This changes the landscape
                        rows[ri] = new_row
                        row_masks[ri] = row_mask
                        used_mask = used_excl | row_mask
                        found_kick = True
                        break
                        
                if found_kick:
                    no_improve_count = 0 # Reset counter after kick
        
        # Update best if we found a better max
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
            
    return best_rows

# EVOLVE-BLOCK-END
