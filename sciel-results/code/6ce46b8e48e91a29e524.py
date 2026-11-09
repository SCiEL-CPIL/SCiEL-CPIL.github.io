import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Helper: Calculate all differences for a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: Check if a row is valid given a set of used differences
    # Returns (is_valid, new_diffs)
    def check_row_validity(row, used_diffs):
        row_diffs = get_diffs(row)
        if row_diffs & used_diffs:
            return False, row_diffs
        return True, row_diffs

    # Helper: Greedy construction of a single row given used_diffs and a target max value (optional)
    # We want to construct a row of length k+1 starting at 0.
    # Heuristic: Pick the smallest valid x that keeps the row "compact"
    def build_row(used_diffs, k, max_limit=None):
        row = [0]
        x = 0
        for _ in range(k):
            x += 1
            if max_limit is not None and x > max_limit:
                return None
            while True:
                if max_limit is not None and x > max_limit:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    # Note: We don't update used_diffs here immediately to allow 
                    # backtracking if we were doing full search, but for greedy 
                    # we commit. However, for re-insertion, we need to ensure 
                    # the whole row is valid against the *external* used_diffs.
                    # The internal diffs are distinct by construction of a Golomb ruler?
                    # No, a row itself must be a Golomb Ruler (distinct internal diffs).
                    # The greedy method above ensures distinct internal diffs?
                    # Let's check: row=[0, 1]. x=2. new_diffs={2, 1}. 
                    # If used_diffs is empty, ok. row=[0,1,2]. 
                    # Internal diffs: 1, 2, 1. Wait. 2-1=1, 1-0=1. Duplicate!
                    # So the simple greedy "smallest x such that no diff with used" 
                    # does NOT guarantee the row itself is a valid Golomb Ruler.
                    # We must ensure internal distinctness too.
                    break
                x += 1
                if max_limit is not None and x > max_limit:
                    return None
            # We need to ensure the row itself has distinct differences.
            # The standard greedy for Golomb Rulers is:
            # x = smallest integer > row[-1] such that all (x - a_i) are distinct 
            # from each other and from used_diffs.
            
        return row

    # Corrected Greedy Builder for Golomb Ruler
    def build_golomb_row(used_diffs, k, max_limit=None):
        row = [0]
        x = 0
        for _ in range(k):
            x += 1
            if max_limit is not None and x > max_limit:
                return None
            found = False
            while not found:
                if max_limit is not None and x > max_limit:
                    return None
                # Check against external used_diffs
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    x += 1
                    continue
                # Check internal distinctness
                # The new diffs are x - a_i.
                # They are distinct if a_i are distinct (which they are).
                # But do they collide with existing internal diffs?
                # Existing internal diffs are in 'row_diffs_so_far'
                # We need to maintain the set of diffs for the current row prefix.
                # Let's re-implement carefully.
                # Actually, if we build a Golomb Ruler, the standard greedy is:
                # pick smallest x > last such that x - a_i is not in 
                # (used_diffs U internal_diffs_so_far)
                
                # Let's restart the inner loop with proper state
                # This function is called repeatedly, so let's make it robust.
                pass
            
            # The previous implementation was flawed for internal distinctness.
            # Let's write a clean version.
            pass
        
        # Clean implementation
        row = [0]
        internal_diffs = set()
        x = 0
        for _ in range(k):
            x += 1
            while True:
                if max_limit is not None and x > max_limit:
                    return None
                new_diffs = {x - a for a in row}
                # Check external
                if new_diffs & used_diffs:
                    x += 1
                    continue
                # Check internal
                if new_diffs & internal_diffs:
                    x += 1
                    continue
                # Accept x
                row.append(x)
                internal_diffs |= new_diffs
                break
        return row

    # 1. Initial Solution: Greedy construction
    used = set()
    rows = []
    for i in range(n_rows):
        row = build_golomb_row(used, k)
        if row is None:
            # Fallback: try with a larger limit or random restart
            row = [0]
            x = 0
            for _ in range(k):
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & get_diffs(row)):
                        row.append(x)
                        break
        rows.append(row)
        used |= get_diffs(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local Search: Re-insertion and Compression
    # Strategy: 
    # A. Try to compress existing rows by shifting elements left.
    # B. If stuck, remove a row, and try to rebuild it with a tighter bound.
    
    def calc_scope(rows):
        return max(max(r) for r in rows)

    def try_compress(rows, used_diffs_map):
        """
        Attempts to reduce the scope of individual rows by shifting elements left.
        used_diffs_map: dict from row_idx to set of diffs of that row.
        """
        improved = False
        for r_idx in range(len(rows)):
            row = rows[r_idx]
            # We want to minimize max(row).
            # Try to shift elements from right to left.
            # Changing row[i] affects diffs (row[j] - row[i]) for j>i and (row[i] - row[j]) for j<i.
            # It's safer to shift from the rightmost element leftwards.
            
            # Get diffs of this row excluding itself from the global set?
            # No, we need to check against ALL other rows' diffs.
            other_diffs = set()
            for i, r in enumerate(rows):
                if i != r_idx:
                    other_diffs |= get_diffs(r)
            
            # Try to reduce the last element first
            # Then the second last, etc.
            # This is a bit slow if done naively. 
            # Let's try a simple pass: for each position p from k down to 1:
            # try to decrease row[p] by 1, 2, ... until valid.
            
            current_row = row[:]
            for p in range(k, 0, -1):
                orig_val = current_row[p]
                min_val = current_row[p-1] + 1
                # Try to reduce value
                for new_val in range(orig_val - 1, min_val - 1, -1):
                    current_row[p] = new_val
                    # Check validity of this row against other_diffs
                    row_diffs = get_diffs(current_row)
                    if not (row_diffs & other_diffs):
                        # Valid!
                        row[p] = new_val
                        improved = True
                        # Update other_diffs for subsequent positions?
                        # other_diffs doesn't change, but the row's own diffs do.
                        # We need to be careful: reducing row[p] changes diffs involving p.
                        # Since we are iterating p downwards, and we only check against OTHER rows,
                        # the internal consistency of the row is maintained by the strict increase.
                        # However, we must ensure the row itself is a Golomb Ruler.
                        # get_diffs(current_row) ensures internal distinctness?
                        # No, get_diffs returns a set. If there are duplicates, the set size is smaller.
                        # We need to check if len(row_diffs) == k*(k+1)/2.
                        if len(row_diffs) == k*(k+1)//2:
                            continue
                        else:
                            # Invalid internal structure, revert and try smaller reduction?
                            # Actually, if we just decreased by 1, and it was valid before,
                            # it might break internal distinctness.
                            # Let's just check if it's a valid Golomb Ruler.
                            pass
                    # If not valid, try next smaller value? 
                    # No, if val-1 is invalid, val-2 might be valid? 
                    # Unlikely for small steps, but possible.
                    # For efficiency, if val-1 fails, we can stop trying this position 
                    # if we assume monotonicity, but it's not strictly monotonic.
                    # Let's just try val-1 and val-2.
                    if new_val == orig_val - 1:
                        # Try one more step down
                        current_row[p] = new_val - 1
                        if current_row[p] >= min_val:
                            row_diffs2 = get_diffs(current_row)
                            if not (row_diffs2 & other_diffs) and len(row_diffs2) == k*(k+1)//2:
                                row[p] = new_val - 1
                                improved = True
                                break
                    break
            # Re-update global used_diffs_map if improved
            if improved:
                used_diffs_map[r_idx] = get_diffs(row)
        
        return improved

    # Main Loop
    time_limit = deadline
    while time.time() < time_limit:
        # 1. Attempt Compression
        # Build a map of diffs for each row
        diffs_map = {i: get_diffs(rows[i]) for i in range(n_rows)}
        
        if try_compress(rows, diffs_map):
            new_scope = calc_scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
            # If we improved, loop again to compress more
            continue
        
        # 2. If no compression, try Re-insertion of a random row
        r_idx = random.randint(0, n_rows - 1)
        old_row = rows[r_idx]
        
        # Remove this row's diffs from the global set
        all_other_diffs = set()
        for i in range(n_rows):
            if i != r_idx:
                all_other_diffs |= diffs_map[i]
        
        # Try to rebuild this row with a tighter bound
        # Current max entry of this row is old_row[-1]
        # Try to build with max_limit = old_row[-1] - 1
        # If that fails, try -2, etc., down to some limit
        success = False
        current_limit = old_row[-1] - 1
        # Don't go below the theoretical minimum or too low
        min_possible = k + 1 # Very rough lower bound
        
        while current_limit >= min_possible and time.time() < time_limit:
            new_row = build_golomb_row(all_other_diffs, k, max_limit=current_limit)
            if new_row is not None:
                # Verify it's valid (should be by construction)
                rows[r_idx] = new_row
                diffs_map[r_idx] = get_diffs(new_row)
                
                new_scope = calc_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                success = True
                break
            current_limit -= 1
        
        if not success:
            # Restore old row
            rows[r_idx] = old_row
            diffs_map[r_idx] = get_diffs(old_row)
            
            # If we can't improve by re-inserting, try a "Kick"
            # Randomly perturb a row to escape local minima
            if random.random() < 0.5:
                # Pick a random position in a random row and change it
                # This is risky, might break validity.
                # Better: Pick a row, and randomly shift one element by +1 or -2
                # Then try to compress.
                r_idx2 = random.randint(0, n_rows - 1)
                p_idx = random.randint(1, k)
                orig_val = rows[r_idx2][p_idx]
                # Try to move it
                move = random.choice([-2, -1, 1, 2])
                new_val = orig_val + move
                min_v = rows[r_idx2][p_idx-1] + 1
                max_v = rows[r_idx2][p_idx+1] - 1 if p_idx < k else orig_val + 5
                
                if min_v <= new_val <= max_v:
                    rows[r_idx2][p_idx] = new_val
                    # Check validity
                    row_diffs = get_diffs(rows[r_idx2])
                    valid = len(row_diffs) == k*(k+1)//2
                    if valid:
                        other_diffs = set()
                        for i in range(n_rows):
                            if i != r_idx2:
                                other_diffs |= diffs_map[i]
                        if not (row_diffs & other_diffs):
                            diffs_map[r_idx2] = row_diffs
                            # Try to compress immediately
                            if try_compress(rows, diffs_map):
                                new_scope = calc_scope(rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    best_rows = [r[:] for r in rows]
                        else:
                            # Revert
                            rows[r_idx2][p_idx] = orig_val
                    else:
                        # Revert
                        rows[r_idx2][p_idx] = orig_val

    return best_rows
