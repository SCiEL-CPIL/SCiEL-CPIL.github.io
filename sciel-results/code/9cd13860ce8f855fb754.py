import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Bitmask for used differences
    # Max difference is bounded by scope. 
    # We assume a reasonable upper bound for scope initially.
    # If scope is S, diffs are in [1, S].
    # We can use an integer bitmask where bit d is set if diff d is used.
    
    def get_diffs_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def has_conflict(row, mask):
        m = get_diffs_mask(row)
        return (m & mask) != 0

    def update_mask(mask, row, old_row):
        # Remove old diffs, add new diffs
        # This is slightly expensive if we recompute from scratch, 
        # but for small k it's fine. 
        # Better: XOR out old, XOR in new? No, if diff was unique to old, it becomes 0.
        # If diff was shared (impossible in valid state), it would be wrong.
        # Since we maintain valid states, old diffs are unique to old_row relative to global mask?
        # No, global mask contains diffs from ALL rows.
        # So: new_mask = (mask & ~get_diffs_mask(old_row)) | get_diffs_mask(row)
        
        old_mask = get_diffs_mask(old_row)
        new_mask = get_diffs_mask(row)
        return (mask & ~old_mask) | new_mask

    # 1. Greedy Construction
    def greedy_solve():
        rows = []
        used_mask = 0
        for r in range(n_rows):
            row = [0]
            for c in range(1, k + 1):
                found = False
                # Try small increments first
                # Heuristic: try values that keep diffs small
                last_val = row[-1]
                # We need to find x > last_val such that {x - y for y in row} are not in used_mask
                # And also not in current row's diffs (which are implicit in the check against used_mask if we update it? No, used_mask is global)
                # Wait, if we add x, the new diffs are {x-y}. We must check if any are in used_mask.
                # Also, we must ensure x-y are distinct within the row? 
                # If y1 != y2, x-y1 != x-y2. So distinctness within row is automatic for strict increase.
                
                # Search range: last_val + 1 to last_val + 100? 
                # If greedy fails, we might need to backtrack, but let's try a simple greedy first.
                # To be safer, we can use a randomized greedy with restarts.
                
                # For speed, let's just try to find a valid x.
                # If we can't find one quickly, we might need a smarter search.
                # But for n_rows=4, k=7, greedy usually works if we pick "small" valid numbers.
                
                # Let's try a window of candidates
                max_try = 200
                for x in range(last_val + 1, last_val + 1 + max_try):
                    new_diffs_mask = 0
                    valid = True
                    for y in row:
                        d = x - y
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_mask |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used_mask |= new_diffs_mask
                        found = True
                        break
                
                if not found:
                    # Greedy stuck. Return None to signal failure
                    return None
            rows.append(row)
        
        return rows

    # 2. Local Search / Polish
    def polish(rows):
        # Iteratively try to reduce the max element of any row
        # by shifting elements or swapping? 
        # Shifting is easier: change one element a_j to a_j' > a_{j-1} and < a_{j+1} (if exists)
        # such that all new diffs are valid.
        
        # We want to minimize the global max.
        # Let's identify the row with the current global max.
        # Try to reduce it.
        
        current_rows = [row[:] for row in rows]
        current_max = max(max(row) for row in current_rows)
        used_mask = 0
        for row in current_rows:
            used_mask |= get_diffs_mask(row)
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Find the row with the largest last element
            max_row_idx = -1
            max_val = -1
            for i, row in enumerate(current_rows):
                if row[-1] > max_val:
                    max_val = row[-1]
                    max_row_idx = i
            
            if max_row_idx == -1:
                break
                
            row = current_rows[max_row_idx]
            
            # Try to reduce the last element first, then others?
            # Or try to reduce any element that allows a smaller max?
            # Simple strategy: Try to move the last element to a smaller value.
            # But it must be > row[-2].
            
            # Let's try to find a new value for the last element that is smaller than current.
            # Range: row[-2] + 1 to row[-1] - 1
            
            old_last = row[-1]
            found_better = False
            
            # Try values in descending order from old_last - 1 down to row[-2] + 1
            # This might be slow if range is large. 
            # Let's limit the search range or use random sampling?
            # Given k=7, the gaps are small. 
            # Let's try a window of 50 values below old_last.
            
            search_range = 100
            low = max(row[-2] + 1, old_last - search_range)
            
            for new_last in range(old_last - 1, low - 1, -1):
                # Check validity
                new_diffs_mask = 0
                valid = True
                for y in row[:-1]:
                    d = new_last - y
                    # Check if this diff is used by OTHER rows
                    # We need to remove the old diffs of this row from used_mask first?
                    # Or check against (used_mask & ~get_diffs_mask(row))
                    
                    # Let's precompute mask of other rows
                    # For efficiency, let's just check if d is in used_mask.
                    # If d is in used_mask, it might be from this row's old state.
                    # So we must exclude this row's current diffs.
                    
                    # Optimization: Precompute 'other_mask' = used_mask & ~get_diffs_mask(row)
                    # Then check if (other_mask >> d) & 1 is 0.
                    
                    pass # Logic inside loop below
            
            # Let's restructure the check for efficiency
            row_mask = get_diffs_mask(row)
            other_mask = used_mask & ~row_mask
            
            for new_last in range(old_last - 1, low - 1, -1):
                valid = True
                new_diffs_mask = 0
                for y in row[:-1]:
                    d = new_last - y
                    if (other_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    # Check if new_last creates conflicts within the row?
                    # No, because row[:-1] is fixed and strictly increasing, 
                    # and new_last > row[-2]. 
                    # Diffs new_last - y are distinct for distinct y.
                    # So we just need to ensure they don't conflict with OTHER rows.
                    
                    # Update state
                    current_rows[max_row_idx][-1] = new_last
                    used_mask = (used_mask & ~row_mask) | new_diffs_mask
                    improved = True
                    found_better = True
                    break
            
            if not found_better:
                # Try to modify other elements? 
                # Or just break if we can't improve the last element of the max row?
                # Maybe try to swap rows? No, rows are indistinguishable for the score (max of all).
                # If we can't reduce the max of the current max row, maybe we are stuck?
                # We could try to reduce the second to last element to allow a smaller last element?
                # That's complex. 
                # Let's add a simple random restart/shake if stuck?
                # Or just accept this local optimum.
                
                # To avoid infinite loop if no improvement, we break.
                # But maybe we can try to reduce a different row?
                # The global max is the bottleneck. If we can't reduce it, we can't improve score.
                # Unless reducing a non-max row allows a future reduction? 
                # Unlikely to help the global max directly.
                
                # Let's try one more step: Randomly perturb a non-max element and see if it helps?
                # For now, break to avoid wasting time.
                break

        return current_rows

    # Run Greedy
    best_sol = greedy_solve()
    if best_sol is None:
        # Fallback: Simple random restart greedy
        for _ in range(10):
            random.seed(int(time.time() * 1000) % 10000)
            best_sol = greedy_solve()
            if best_sol:
                break
    
    if best_sol is None:
        # Extreme fallback: Return a trivial valid set (might be large)
        # Construct a set that is definitely valid but large
        # e.g. powers of 2?
        # Row 0: 0, 1, 2, ...
        # This might not be valid if diffs overlap.
        # Let's just return a set of zeros? No, must be increasing.
        # Let's construct a "safe" set with large gaps.
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            for c in range(1, k+1):
                x = 1
                while True:
                    diffs = {x - y for y in row}
                    if not (diffs & used):
                        row.append(x)
                        used |= diffs
                        break
                    x += 1
            rows.append(row)
        best_sol = rows

    # Polish
    if time.time() < deadline - 2.0:
        polished = polish(best_sol)
        if polished:
            # Verify validity of polished
            # (Assume polish maintains validity, but double check is cheap for small k)
            used_mask = 0
            valid = True
            for row in polished:
                m = get_diffs_mask(row)
                if m & used_mask:
                    valid = False
                    break
                used_mask |= m
            
            if valid:
                best_sol = polished

    return best_sol
