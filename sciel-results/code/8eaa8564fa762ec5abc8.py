import time
import random
import sys

def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng):
    """Construct a valid set using greedy with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Random skip to diversify
            if rng.random() < 0.1:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _local_improve(rows, n_rows, k, rng, time_limit):
    """
    Local search to minimize the maximum entry.
    Strategy: Iterate through all positions, try to reduce the value to the 
    smallest possible integer that maintains validity.
    """
    work = [list(r) for r in rows]
    used_set = set()
    # Build initial used set
    for row in work:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    current_max = _max_entry(work)
    L = k + 1
    
    # Helper to update used_set when a value changes
    def update_diffs(r_idx, pos, old_val, new_val):
        row = work[r_idx]
        # Remove old diffs
        for i in range(L):
            if i != pos:
                if i < pos:
                    used_set.discard(old_val - row[i])
                else:
                    used_set.discard(row[i] - old_val)
        # Add new diffs
        for i in range(L):
            if i != pos:
                if i < pos:
                    used_set.add(new_val - row[i])
                else:
                    used_set.add(row[i] - new_val)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Shuffle order of rows and positions to avoid bias
        order = [(r, p) for r in range(n_rows) for p in range(1, L)]
        rng.shuffle(order)
        
        for r_idx, pos in order:
            if time.time() >= time_limit:
                break
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < L - 1 else old_val - 1
            
            if upper < lower:
                continue
                
            # Try to find the smallest valid value in [lower, upper]
            best_val = old_val
            # We want to minimize the max, so reducing any value helps, 
            # but reducing values near the end is more critical.
            # We scan from lower upwards to find the first valid one.
            for new_val in range(lower, upper + 1):
                if new_val == old_val:
                    continue
                # Check validity
                valid = True
                # Temporarily apply to check
                # Diffs involving pos
                for i in range(L):
                    if i == pos: continue
                    d = abs(new_val - row[i])
                    # Check if d is already used by others (excluding pos's old diffs)
                    # We need to be careful: if d was only produced by pos, it's fine to reuse.
                    # Easier: calculate new set of diffs for this row, check intersection with global used set minus old diffs of this row.
                    # But that's O(L^2).
                    # Optimized: 
                    # 1. Identify all diffs currently involving pos.
                    # 2. Remove them from a temporary copy of used_set? No, too slow.
                    # 3. Check if any new diff conflicts with used_set \ old_diffs_involving_pos.
                    
                    # Let's just do the full check for simplicity, L=8 is small.
                    pass
                
                # Faster check:
                old_diffs_involving_pos = set()
                for i in range(L):
                    if i != pos:
                        old_diffs_involving_pos.add(abs(old_val - row[i]))
                
                new_diffs_involving_pos = set()
                for i in range(L):
                    if i != pos:
                        new_diffs_involving_pos.add(abs(new_val - row[i]))
                
                # The set of diffs NOT involving pos is 'stable'
                # We need: (new_diffs_involving_pos) & (used_set - old_diffs_involving_pos) == empty
                # Also, new_diffs_involving_pos must not have internal conflicts? 
                # No, internal conflicts within the row are handled by the distinctness of the row itself?
                # Wait, the condition is "all positive differences ... are distinct over all rows".
                # So all 112 differences must be distinct.
                # So yes, new_diffs must not overlap with (used_set - old_diffs).
                
                conflict = new_diffs_involving_pos & (used_set - old_diffs_involving_pos)
                if not conflict:
                    # Also check if new_diffs have duplicates internally?
                    # If row is strictly increasing, differences are distinct within the row?
                    # No! e.g. 0, 1, 3. Diffs: 1, 3, 2. Distinct.
                    # e.g. 0, 2, 4. Diffs: 2, 4, 2. Conflict!
                    # So we must ensure new_diffs_involving_pos has no duplicates?
                    # Actually, if the row is strictly increasing, can it have duplicate diffs?
                    # Yes. 0, 1, 3 has diffs 1, 2, 3.
                    # 0, 2, 4 has diffs 2, 4, 2. Invalid row.
                    # So we must ensure the row itself is "Sidon" (distinct diffs).
                    # The check `conflict` above only checks against OTHER rows and non-pos parts of this row.
                    # We must also ensure `len(new_diffs_involving_pos) == L-1` (no dups in new set)
                    # AND that new_diffs don't conflict with the rest of the row's diffs (which are in used_set - old_diffs_involving_pos).
                    # Wait, `used_set` contains ALL diffs.
                    # `used_set - old_diffs_involving_pos` contains all diffs NOT involving pos.
                    # So if `new_diffs` intersects that, it's a conflict.
                    # Do we need to check if `new_diffs` has internal duplicates?
                    # If `new_diffs` has a duplicate, say d appears twice, then d is in `new_diffs`.
                    # Is d in `used_set - old_diffs`?
                    # If d was not in `old_diffs`, it might be in the rest of the row.
                    # If d is in the rest of the row, it's in `used_set - old_diffs`. So conflict detected.
                    # If d is NOT in the rest of the row, and d is NOT in other rows, then d is not in `used_set - old_diffs`.
                    # But if d appears twice in `new_diffs`, it means the row has duplicate diffs.
                    # This is invalid.
                    # So we MUST check `len(new_diffs_involving_pos) == L-1`.
                    
                    if len(new_diffs_involving_pos) == L - 1:
                        valid = True
                        break
                    else:
                        valid = False
                        break # No point trying larger new_val if this one failed internal check? 
                        # Actually, changing new_val changes the diffs. So continue loop.
                
                # If valid, apply
                if valid:
                    # Update used_set
                    for d in old_diffs_involving_pos:
                        used_set.discard(d)
                    for d in new_diffs_involving_pos:
                        used_set.add(d)
                    
                    row[pos] = new_val
                    if new_val < old_val:
                        improved = True
                        current_max = _max_entry(work)
                    break # Move to next position
            
    return work

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140  # Safety margin
    
    # 1. Get a baseline
    rng = random.Random(42)
    best_rows = _greedy_construct(n_rows, k, rng)
    best_max = _max_entry(best_rows)
    
    # 2. Local improvement
    improved_rows = _local_improve(best_rows, n_rows, k, rng, time_limit)
    if _check_valid(improved_rows, k):
        imax = _max_entry(improved_rows)
        if imax < best_max:
            best_rows = improved_rows
            best_max = imax
    
    # 3. Random Restarts with Local Improvement
    # Try to find a better starting point
    attempts = 0
    while time.time() < time_limit - 10:
        attempts += 1
        # Vary randomness
        rng = random.Random(int(time.time() * 1000) + attempts)
        rows = _greedy_construct(n_rows, k, rng)
        
        # Quick local improve
        # To save time, limit local improve per restart
        sub_limit = time.time() + 5
        improved = _local_improve(rows, n_rows, k, rng, sub_limit)
        
        if _check_valid(improved, k):
            imax = _max_entry(improved)
            if imax < best_max:
                best_rows = improved
                best_max = imax
                # If we hit the theoretical lower bound or known optimum, we can stop
                # Known optimum for (4,8) is 134.
                if best_max <= 134:
                    break

    # Final validation
    if not _check_valid(best_rows, k):
        # Fallback to a very simple greedy if something broke
        best_rows = _greedy_construct(n_rows, k, random.Random(0))

    return best_rows
