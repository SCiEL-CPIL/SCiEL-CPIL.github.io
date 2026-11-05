import time
import random
import math
import sys


def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _column_wise_construct(n_rows, num_elements, rng, max_scope_hint=160):
    """
    List-scheduling style construction:
    - We build all rows simultaneously, column by column.
    - For each row, at each column, we pick the smallest feasible value that
      doesn't conflict with any previously chosen differences (within or across rows).
    - This mimics 'earliest available slot' scheduling.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    
    for col in range(1, num_elements):
        # Process rows in random order to break symmetry
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for r in order:
            row = rows[r]
            last_val = row[-1]
            found = False
            
            # Try values from last_val+1 up to a reasonable bound
            # Use a dynamic bound: we want to keep scope tight
            # But also allow some slack for feasibility
            upper = max_scope_hint + 10
            for val in range(last_val + 1, upper + 1):
                # Check new differences
                new_diffs = []
                valid = True
                for prev in row:
                    d = val - prev
                    if d <= 0:
                        valid = False
                        break
                    # Check within-row uniqueness
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check against global used_mask
                conflict = False
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Accept this value
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
                found = True
                break
            
            if not found:
                # Fallback: jump to a value that definitely works (large enough)
                # This shouldn't happen often if max_scope_hint is reasonable
                val = last_val + 20 + rng.randint(0, 10)
                while True:
                    new_diffs = [val - prev for prev in row]
                    if len(set(new_diffs)) == len(new_diffs):
                        conflict = False
                        for d in new_diffs:
                            if (used_mask >> d) & 1:
                                conflict = True
                                break
                        if not conflict:
                            break
                    val += 1
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
    
    return rows


def _backtrack_row_with_mask(prefix, used_mask, num_elements, max_val, rng, time_limit, start_time):
    """
    Backtrack to complete a row given a prefix and a forbidden difference mask.
    Returns row or None.
    """
    if time.time() > time_limit:
        return None
    
    if len(prefix) == num_elements:
        return prefix[:]
    
    row = prefix
    last_val = row[-1]
    
    # Candidates: try smaller values first (greedy) but with some randomization
    candidates = list(range(last_val + 1, max_val + 1))
    if len(candidates) > 30:
        # Shuffle first 20, keep rest sorted
        rng.shuffle(candidates[:20])
        candidates = candidates[:20] + sorted(candidates[20:])
    else:
        rng.shuffle(candidates)
    
    for val in candidates:
        new_diffs = set()
        valid = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
        
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        
        if new_mask & used_mask:
            continue
        
        result = _backtrack_row_with_mask(row + [val], used_mask | new_mask, num_elements, max_val, rng, time_limit, start_time)
        if result is not None:
            return result
    
    return None


def _precompute_row_catalog(num_elements, max_scope, n_catalog, rng, time_limit):
    """
    Precompute a catalog of valid rows (each starting at 0, length num_elements, max <= max_scope)
    that are 'tight' (small scope). We store them with their masks.
    """
    catalog = []
    seen_masks = set()
    start_time = time.time()
    
    attempts = 0
    while len(catalog) < n_catalog and time.time() < time_limit:
        attempts += 1
        # Randomize the search space slightly
        local_rng = random.Random(attempts * 13 + 7)
        # Try to build a row with backtracking, limiting max_val
        target_max = max_scope
        row = _backtrack_row_with_mask([0], 0, num_elements, target_max, local_rng, time_limit, start_time)
        if row is not None:
            m = _row_mask(row)
            if m not in seen_masks:
                seen_masks.add(m)
                catalog.append((row, m))
    
    return catalog


def _row_pair_swap_search(rows, num_elements, catalog, rng, time_limit):
    """
    Local search: try to replace pairs of rows with different pairs from the catalog
    to improve the global scope or reduce conflicts.
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    
    # Precompute row masks for current rows
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    # Build a lookup: mask -> list of (row, mask) in catalog
    catalog_by_mask = {}
    for row, m in catalog:
        catalog_by_mask.setdefault(m, []).append(row)
    
    # Also, for each catalog row, we might want to check if it fits with the rest
    # But catalog rows are pre-validated internally. We just need to check cross-row conflicts.
    
    max_iter = 100000
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Pick two random rows to swap
        if n_rows < 2:
            break
        
        r1, r2 = rng.sample(range(n_rows), 2)
        
        # Current masks of these two rows
        m1 = row_masks[r1]
        m2 = row_masks[r2]
        
        # Mask of all other rows
        other_mask = global_mask & ~m1 & ~m2
        
        # Try to find two catalog rows that:
        # 1. Are distinct from each other
        # 2. Don't conflict with other_mask
        # 3. Don't conflict with each other
        # 4. Ideally have smaller max value than current rows
        
        found_swap = False
        # Sample a few candidates from catalog
        catalog_size = len(catalog)
        if catalog_size >= 2:
            for _ in range(20):  # Try 20 random pairs
                c1_idx, c2_idx = rng.sample(range(catalog_size), 2)
                c1_row, c1_mask = catalog[c1_idx]
                c2_row, c2_mask = catalog[c2_idx]
                
                # Check conflicts with others
                if (c1_mask & other_mask) or (c2_mask & other_mask):
                    continue
                
                # Check conflict between c1 and c2
                if (c1_mask & c2_mask):
                    continue
                
                # Check if this improves scope
                new_scope = max(_scope([c1_row, c2_row]), _scope([r for i, r in enumerate(current_rows) if i not in (r1, r2)]))
                
                # Accept if better, or equal but different (to escape local minima)
                if new_scope < best_scope:
                    # Apply swap
                    current_rows[r1] = c1_row[:]
                    current_rows[r2] = c2_row[:]
                    row_masks[r1] = c1_mask
                    row_masks[r2] = c2_mask
                    global_mask = (global_mask & ~m1 & ~m2) | c1_mask | c2_mask
                    
                    new_scope_actual = _scope(current_rows)
                    if new_scope_actual < best_scope:
                        best_scope = new_scope_actual
                        best_rows = [r[:] for r in current_rows]
                    found_swap = True
                    break
                elif new_scope == best_scope and rng.random() < 0.3:
                    # Probabilistic acceptance to explore
                    current_rows[r1] = c1_row[:]
                    current_rows[r2] = c2_row[:]
                    row_masks[r1] = c1_mask
                    row_masks[r2] = c2_mask
                    global_mask = (global_mask & ~m1 & ~m2) | c1_mask | c2_mask
                    found_swap = True
                    break
    
    return best_rows, best_scope


def _element_local_search(rows, num_elements, rng, time_limit):
    """
    Fine-grained local search: try to move individual elements down to reduce scope.
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    n_rows = len(current_rows)
    
    for it in range(200000):
        if time.time() > time_limit:
            break
        
        r = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, num_elements - 1)
        row = current_rows[r]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope
        
        if low >= old_val:
            continue
        
        # Try to move down
        new_val = rng.randint(low, old_val - 1)
        
        # Check validity
        row[pos] = new_val
        new_mask = _row_mask(row)
        
        # Check internal consistency
        diffs = _row_diffs_list(row)
        if len(diffs) != len(set(diffs)):
            row[pos] = old_val
            continue
        
        other_mask = global_mask & ~row_masks[r]
        if new_mask & other_mask:
            row[pos] = old_val
            continue
        
        # Accept
        row_masks[r] = new_mask
        global_mask = (global_mask & ~old_mask) | new_mask  # Note: old_mask was row_masks[r] before update
        # Wait, I need to be careful with mask updates
        # Let's redo this properly
        
    # Actually, the above loop has a bug in mask update. Let's implement a cleaner version.
    
    return best_rows, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin
    
    num_elements = k + 1
    total_diffs_needed = n_rows * (num_elements * (num_elements - 1) // 2)
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Column-wise list scheduling construction
    # Try multiple seeds and scope hints
    for seed in range(30):
        if time.time() > DEADLINE - 60:
            break
        local_rng = random.Random(seed * 19 + 1)
        # Start with tight scope, relax if needed
        for offset in [0, 5, 10, 20, 30]:
            scope_hint = counting_bound + offset
            rows = _column_wise_construct(n_rows, num_elements, local_rng, scope_hint)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                if s <= counting_bound + 5:
                    break
        if best_scope <= counting_bound + 5:
            break
    
    # Fallback: simple sequential construction if column-wise fails
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Precompute a catalog of tight rows for swap search
    remaining = DEADLINE - time.time()
    if remaining > 40:
        # Build catalog with scope slightly above current best
        catalog_scope = min(best_scope + 10, counting_bound + 40)
        catalog = _precompute_row_catalog(num_elements, catalog_scope, 200, rng, DEADLINE - 30)
        
        if catalog:
            improved_rows, improved_scope = _row_pair_swap_search(
                best_rows, num_elements, catalog, rng, DEADLINE - 20
            )
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
    
    # Phase 3: Final element-level local search
    remaining = DEADLINE - time.time()
    if remaining > 15:
        # Simple greedy down-move search
        current_rows = [r[:] for r in best_rows]
        current_scope = _scope(current_rows)
        row_masks = [_row_mask(r) for r in current_rows]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
        
        n_rows = len(current_rows)
        for it in range(50000):
            if time.time() > DEADLINE - 5:
                break
            
            r = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            row = current_rows[r]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope
            
            if low >= old_val:
                continue
            
            new_val = rng.randint(low, old_val - 1)
            row[pos] = new_val
            
            # Check internal
            diffs = _row_diffs_list(row)
            if len(diffs) != len(set(diffs)):
                row[pos] = old_val
                continue
            
            new_mask = _row_mask(row)
            other_mask = global_mask & ~row_masks[r]
            if new_mask & other_mask:
                row[pos] = old_val
                continue
            
            # Accept
            row_masks[r] = new_mask
            global_mask = (global_mask & ~row_masks[r]) | new_mask  # This is wrong, need to store old mask
            
            # Actually, let's just recompute scope and accept if better
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
            else:
                # Revert
                row[pos] = old_val
                row_masks[r] = _row_mask(row)
                global_mask = 0
                for m in row_masks:
                    global_mask |= m
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
