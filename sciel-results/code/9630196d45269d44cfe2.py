import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

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

def _construct_row(k, forbidden_diffs, max_val):
    """
    Attempt to construct a single row of k+1 integers starting at 0,
    strictly increasing, max <= max_val,
    such that all internal differences are distinct and not in forbidden_diffs.
    Returns the row if found, None otherwise.
    Uses backtracking with forward checking.
    """
    row = [0]
    
    def backtrack(idx, current_row, current_diffs):
        if idx == k:
            # Need one more element to reach size k+1?
            # Wait, row size is k+1. Indices 0..k.
            # If idx is the index of the next element to add.
            # We start with row=[0], idx=1.
            # We want to add elements at indices 1..k.
            # So base case is when len(current_row) == k+1.
            pass
        
        # Check if we have enough room
        if len(current_row) == k + 1:
            return True
        
        last_val = current_row[-1]
        # The next value must be > last_val and <= max_val
        # Also, the difference with last_val must be valid.
        
        # Heuristic: Try values that create small differences first?
        # Or values that are "tight" to the previous?
        
        # Determine range for next value
        start_val = last_val + 1
        end_val = min(max_val, start_val + 50) # Limit search width for speed?
        # Actually, max_val is small (105), so we can iterate up to max_val.
        
        # To improve speed, we can order candidates.
        # Candidates: start_val to max_val
        # But we must ensure that the difference d = candidate - last_val is not in current_diffs or forbidden_diffs.
        
        for next_val in range(start_val, max_val + 1):
            d = next_val - last_val
            if d in current_diffs or d in forbidden_diffs:
                continue
            
            # Check if adding next_val creates any other conflict with previous elements?
            # We only need to check d = next_val - last_val if we maintain current_diffs correctly?
            # No, next_val - current_row[i] for all i < len(current_row) must be unique and not forbidden.
            
            new_diffs_to_check = {next_val - a for a in current_row}
            
            # Check against current_diffs (internal)
            if new_diffs_to_check & current_diffs:
                continue
            # Check against forbidden_diffs (external)
            if new_diffs_to_check & forbidden_diffs:
                continue
            
            # Also check if new_diffs_to_check has internal duplicates (which is impossible as current_row is unique)
            
            # Pruning: Check if it's still possible to complete the row
            # Remaining slots: k + 1 - (len(current_row) + 1)
            # Min possible next diff: 1
            # Max possible next diff: max_val - next_val
            # This is complex. Let's rely on the tight max_val bound.
            
            current_row.append(next_val)
            current_diffs |= new_diffs_to_check
            
            if backtrack(idx + 1, current_row, current_diffs):
                return True
            
            current_row.pop()
            current_diffs -= new_diffs_to_check
            
        return False

    # Start backtracking
    # We need to handle the first step specially or just let the loop handle it.
    # current_row = [0], current_diffs = set()
    # idx is not really used in the loop above, I used len(current_row).
    
    if backtrack(0, row, set()):
        return row
    return None

def _try_improve_row(rows, ri, max_allowed, forbidden_diffs):
    """
    Try to reconstruct row ri to have max entry <= max_allowed.
    forbidden_diffs is the set of differences used by all other rows.
    Returns the new row if successful, else None.
    """
    # Add some randomness to the search order in _construct_row?
    # The current _construct_row is deterministic (tries smallest first).
    # We might want to randomize the candidate order in _construct_row.
    # Let's modify _construct_row to accept a randomization flag or just shuffle candidates.
    
    # For now, we'll rely on multiple calls or slight variations.
    # To make it more effective, we can shuffle the range in _construct_row.
    # But _construct_row is called frequently.
    
    # Let's create a randomized version of the row construction.
    row = [0]
    
    def backtrack_randomized(idx, current_row, current_diffs):
        if len(current_row) == k_global + 1:
            return True
        
        last_val = current_row[-1]
        start_val = last_val + 1
        end_val = min(max_allowed, start_val + 100) # Cap to avoid huge loops if max_allowed is large
        # Actually max_allowed is small.
        
        candidates = list(range(start_val, max_allowed + 1))
        random.shuffle(candidates)
        
        for next_val in candidates:
            d = next_val - last_val
            if d in current_diffs or d in forbidden_diffs:
                continue
            
            new_diffs_to_check = {next_val - a for a in current_row}
            
            if new_diffs_to_check & current_diffs:
                continue
            if new_diffs_to_check & forbidden_diffs:
                continue
            
            current_row.append(next_val)
            current_diffs |= new_diffs_to_check
            
            if backtrack_randomized(idx + 1, current_row, current_diffs):
                return True
            
            current_row.pop()
            current_diffs -= new_diffs_to_check
            
        return False

    if backtrack_randomized(0, row, set()):
        return row
    return None

def construct(n_rows, k):
    global k_global
    k_global = k
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    best_rows = None
    best_val = float('inf')
    
    # Try a few greedy inits
    for _ in range(10):
        rows = _greedy_init(n_rows, k)
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Precompute diffs for all rows
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    # 2. Iterated Local Search with Row Reconstruction
    # Strategy: Pick a row, try to reconstruct it with a smaller max.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Check if we hit the theoretical lower bound
        # Lower bound is roughly (n_rows * k*(k+1)/2) / k ? No.
        # Total diffs = n_rows * k*(k+1)/2.
        # Max diff >= Total diffs.
        # For n=7, k=6: Total = 7 * 15 = 105.
        # So Max >= 105.
        if current_val <= 105:
            break
            
        # Pick a random row to improve
        ri = random.randrange(n_rows)
        
        # Calculate forbidden diffs (all diffs except row ri)
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Try to reconstruct row ri with max_allowed = current_val - 1
        # If that fails, try current_val - 2, etc?
        # Or just try current_val - 1. If it fails, we might try a random perturbation.
        
        new_row = None
        # Try a few times with randomization
        for attempt in range(20):
            new_row = _try_improve_row(current_rows, ri, current_val - 1, forbidden)
            if new_row:
                break
        
        if new_row:
            # Validate the new row (should be valid by construction, but double check)
            new_diffs = _row_diffs(new_row)
            # Check no overlap with forbidden
            if new_diffs & forbidden:
                # Should not happen, but safety
                new_row = None
            else:
                # Update
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                current_val = _global_max(current_rows)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
            # If stuck, perturb the current solution
            if no_improve_count > max_no_improve:
                # Perturb: Randomly change a row
                pi = random.randrange(n_rows)
                # Reconstruct row pi with a larger max to allow variation
                forbidden = set()
                for j in range(n_rows):
                    if j != pi:
                        forbidden |= row_diffs[j]
                
                new_row = _try_improve_row(current_rows, pi, current_val + 10, forbidden)
                if new_row:
                    current_rows[pi] = new_row
                    row_diffs[pi] = _row_diffs(new_row)
                    current_val = _global_max(current_rows)
                    # Do not update best yet, just restart search
                no_improve_count = 0
                
    return best_rows

# EVOLVE-BLOCK-END
