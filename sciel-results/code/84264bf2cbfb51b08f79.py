<<<<<<< SEARCH
def _difference_swap_mutation(rows, k, used, num_swaps=None, rng=None):
    """Perform difference-swap moves to reduce the max entry."""
    if rng is None:
        rng = random.Random()
    if num_swaps is None:
        num_swaps = rng.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            continue
        
        best_new_val = None
        best_score = float('inf')
        
        candidates = set()
        for _ in range(30):
            candidates.add(rng.randint(low, high))
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 8):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val or new_val < low or new_val > high:
                continue
            
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used
=======
def _regenerate_row(rows, k, target_scope, rng, deadline):
    """Try to regenerate a specific row to reduce the global scope."""
    if time.time() > deadline:
        return rows
    
    # Pick the row with the largest max element
    max_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
    old_row = rows[max_idx]
    old_max = max(old_row)
    
    if old_max <= target_scope:
        return rows
    
    # Remove old row's differences
    used = _all_diffs_bitset(rows)
    old_diffs = _row_diffs_bitset(old_row)
    used_without_old = used & ~old_diffs
    
    # Try to find a new row with max <= target_scope
    # Use backtracking with the current used set
    new_row = _backtrack_row_bitset(k, used_without_old, target_scope, rng, deadline)
    
    if new_row is not None:
        rows[max_idx] = new_row
        return rows
    
    return rows
>>>>>>> REPLACE

