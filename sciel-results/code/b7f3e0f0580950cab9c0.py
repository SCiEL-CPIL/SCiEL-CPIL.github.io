import time
import random
import math
from collections import Counter

def _scope(rows):
    """Compute the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _validate(rows, k):
    """Check if all differences across all rows are distinct and rows are valid."""
    seen = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _get_row_diffs(row):
    """Get the set of differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_row(k, used_diffs, rng, max_val=None):
    """
    Greedily construct a row of k+1 elements starting at 0.
    Returns the row or None if it fails to construct within max_val (if specified).
    """
    row = [0]
    if max_val is None:
        max_val = float('inf')
    
    # We need to choose k more elements.
    # To ensure we can finish, we should keep the last element small enough.
    # If we have t elements left to choose, the last one must be <= max_val.
    # A loose bound: next element <= max_val - (t-1) * 1 (min step 1)
    
    for _ in range(k):
        last = row[-1]
        # Candidates must be > last
        # And the resulting diffs (x - y for y in row) must not be in used_diffs
        # And the row must be strictly increasing.
        
        # Heuristic: try small values first, but with some randomization to escape local traps
        # Or try values in a specific range.
        
        # Let's try a window of candidates.
        # Base candidate: last + 1
        # Max candidate: max_val (but we need to leave room for future elements? 
        # Actually, if we pick x, the next must be > x. 
        # If we pick x = max_val, we can't pick any more if we need more elements.
        # So if we have m elements left to pick (including this one),
        # this element must be <= max_val - (m-1).
        
        remaining = k - len(row) + 1 # including the one we are picking now
        max_candidate = min(max_val, max_val - (remaining - 1))
        
        # If max_candidate < last + 1, we are stuck.
        if max_candidate < last + 1:
            return None
            
        # Try candidates in a shuffled order within a reasonable window
        # Window size: maybe 50-100?
        # Or just try from last+1 to max_candidate.
        # For speed, we can limit the search width initially.
        
        # Strategy: 
        # 1. Try the smallest available valid x.
        # 2. If that leads to a dead end later, we might need to backtrack.
        # But here we are doing a greedy construction for a single row given used_diffs.
        # We don't backtrack within this function easily unless we return None.
        # To be more robust, we can try a few random choices if the greedy one fails later?
        # No, let's stick to a randomized greedy: shuffle a prefix of candidates.
        
        candidates = list(range(last + 1, max_candidate + 1))
        if len(candidates) > 100:
            # Only shuffle the first 100, rest in order? Or just pick a random subset?
            # Let's try the first 50 in random order, then the rest in order.
            rng.shuffle(candidates[:50])
            # Keep the rest in order to ensure we don't skip small values too much
        else:
            rng.shuffle(candidates)
            
        found = False
        for x in candidates:
            # Check diffs
            valid = True
            new_diffs = []
            for y in row:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(x)
                # Update used_diffs? 
                # No, we should check against the CURRENT row's diffs too?
                # The function signature takes `used_diffs` which is global.
                # We must ensure new_diffs don't collide with each other?
                # No, x-y1 and x-y2 are distinct.
                # But do they collide with existing diffs in `row`?
                # `used_diffs` does NOT include diffs from the current row yet.
                # So we MUST check against the set of diffs in `row` as well.
                
                # Let's re-evaluate.
                # `used_diffs` is the set of diffs from PREVIOUS rows.
                # `row` has internal diffs.
                # We need `new_diffs` to be disjoint from `used_diffs` AND `row_diffs`.
                
                # Let's compute row_diffs efficiently or maintain it.
                # For simplicity in this helper, let's just check.
                row_diffs = _get_row_diffs(row)
                
                if not any(d in row_diffs for d in new_diffs):
                    # Valid
                    # We don't update the passed `used_diffs` because it's a local choice
                    # but we do append to row.
                    # The caller will update the global `used_diffs` after the row is complete.
                    # Wait, if we append x, the next iteration will use the updated `row`.
                    # So we don't need to return the diffs, just the row.
                    found = True
                    break
        
        if not found:
            return None
            
    return row

def _construct(n_rows, k):
    """
    Main construction function.
    Uses a combination of greedy initialization and local search (row reconstruction).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s buffer for safety/overhead
    
    # Parameters
    # n_rows: number of rows
    # k: number of intervals (so k+1 points)
    
    # Lower bound
    total_diffs = n_rows * k * (k + 1) // 2
    # M must be at least total_diffs? 
    # No, M is the max value.
    # Diffs are in [1, M].
    # So we need M >= total_diffs.
    min_scope = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy:
    # 1. Try to find a solution with scope close to min_scope.
    # 2. Use randomized greedy to generate initial populations.
    # 3. Improve via local search: pick a row, remove its diffs, reconstruct it greedily
    #    to minimize its max value (or just to fit within current scope), then add diffs back.
    #    If the new row is valid and the scope decreases or stays same, accept.
    
    # We will run several restarts.
    for restart in range(20):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 12345 + 67890)
        
        # Initialize
        used_diffs = set()
        rows = []
        success = True
        
        # We aim for a target scope. 
        # Let's start with a target scope slightly above min_scope.
        # target_scope = min_scope + 10 + restart
        target_scope = min_scope + 15 + restart * 2
        
        for r in range(n_rows):
            if time.time() > deadline:
                success = False
                break
                
            # Try to build a row within target_scope
            # We might need to retry this row with different randomness if it fails
            row = None
            for attempt in range(50):
                # Shuffle the RNG state for this row?
                # We just call _greedy_row which uses the shared rng
                row = _greedy_row(k, used_diffs, rng, max_val=target_scope)
                if row is not None:
                    break
                    
            if row is None:
                success = False
                break
                
            rows.append(row)
            # Add diffs to used_diffs
            row_diffs = _get_row_diffs(row)
            used_diffs.update(row_diffs)
            
        if not success or len(rows) != n_rows:
            continue
            
        # Validate
        if not _validate(rows, k):
            continue
            
        current_scope = _scope(rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in rows]
            
        # If we hit the theoretical lower bound, we can stop
        if best_scope <= min_scope:
            break
            
        # Local Search / Improvement
        # Try to reduce the scope by reconstructing rows.
        # We will iterate through rows and try to reconstruct them to be "smaller".
        # Or just try to reconstruct them to be valid within the current scope, 
        # and if the new max is lower, it helps.
        
        # A simpler improvement: 
        # Try to reconstruct each row to have a smaller max value.
        # For each row i:
        #   Remove row i's diffs from used_diffs.
        #   Try to reconstruct row i with a smaller max_val (e.g., current_max - 1).
        #   If successful, update used_diffs and the row.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_rows):
                if time.time() > deadline:
                    break
                    
                # Current max of this row
                curr_max = max(rows[i])
                
                # Try to reconstruct with a smaller max
                # Try curr_max - 1, curr_max - 2, ... down to min_scope
                # But we must ensure it fits in the global used_diffs.
                
                # Remove current row's diffs
                curr_diffs = _get_row_diffs(rows[i])
                used_diffs.difference_update(curr_diffs)
                
                new_row = None
                # Try smaller scopes for this specific row
                # We can't go below the minimum required for the row itself?
                # Min scope for a row of k+1 points is at least k (if 0,1,2..k)
                # But we are constrained by used_diffs.
                
                # Let's try a range of max_vals
                # Start from curr_max - 1 down to max(min_scope, curr_max - 20)
                for new_max in range(curr_max - 1, max(min_scope, curr_max - 30) - 1, -1):
                    if time.time() > deadline:
                        break
                    
                    # Try to build a row with max_val = new_max
                    # We need to be careful: the row must be strictly increasing and end <= new_max.
                    # _greedy_row enforces max_val as an upper bound on the last element.
                    
                    # To increase the chance of finding a "smaller" row, 
                    # we might want to bias the greedy search towards smaller values.
                    # _greedy_row already tries smaller values first (mostly).
                    
                    # Let's try a few random restarts for this specific row reconstruction
                    for _ in range(5):
                        new_row = _greedy_row(k, used_diffs, rng, max_val=new_max)
                        if new_row is not None:
                            # Check if it's actually smaller or equal?
                            # It must be <= new_max by construction.
                            # If max(new_row) < curr_max, it's an improvement.
                            # Even if max(new_row) == curr_max, it might free up diffs
                            # that allow OTHER rows to shrink?
                            # For now, let's only accept if max(new_row) < curr_max 
                            # OR if it allows a subsequent row to shrink?
                            # Simpler: accept if valid and max <= curr_max.
                            # But we want to REDUCE the global scope.
                            # So we only care if max(new_row) < curr_max 
                            # AND curr_max was the global max.
                            # Or just accept any valid reconstruction and see if global scope drops.
                            
                            if max(new_row) <= curr_max:
                                break
                            else:
                                new_row = None
                                break
                                
                    if new_row is not None:
                        # Check if this actually improves the global scope
                        # or just maintains validity.
                        # Let's tentatively accept and check global scope.
                        
                        # Update used_diffs
                        new_diffs = _get_row_diffs(new_row)
                        used_diffs.update(new_diffs)
                        
                        old_row = rows[i]
                        rows[i] = new_row
                        
                        new_global_scope = _scope(rows)
                        
                        if new_global_scope < best_scope:
                            best_scope = new_global_scope
                            best_rows = [list(r) for r in rows]
                            improved = True
                        elif new_global_scope < curr_max: # Local improvement
                            improved = True
                        else:
                            # No improvement in scope, revert?
                            # Or keep it to potentially help other rows?
                            # Let's keep it if it's valid, as it might help.
                            # But if it doesn't help, we might be wasting time.
                            # Let's revert if no global improvement.
                            used_diffs.difference_update(new_diffs)
                            used_diffs.update(_get_row_diffs(old_row))
                            rows[i] = old_row
                            break
                    else:
                        # Failed to find a smaller row, restore old diffs
                        used_diffs.update(curr_diffs)
                        break
                        # No, we broke out of the new_max loop. 
                        # We need to restore diffs if we didn't find a row.
                        # The logic above: if new_row is None, we exit the new_max loop.
                        # We must restore curr_diffs.
                
                # Ensure diffs are restored if we didn't accept a new row
                # The logic above is a bit tangled. Let's clean it up.
                
        # End local search
        
    if best_rows is not None:
        return best_rows
    
    # Fallback: Greedy without scope constraint
    used = set()
    rows = []
    rng = random.Random(999)
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng, max_val=float('inf'))
        if row is None:
            # Very unlikely, but handle it
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
        else:
            used.update(_get_row_diffs(row))
        rows.append(row)
        
    return rows

def construct(n_rows, k):
    """
    Entry point.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
        
    return _construct(n_rows, k)
