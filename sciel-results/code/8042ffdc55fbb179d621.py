import time
import random
import math


def _row_diffs_list(row):
    """List of all positive differences within a row."""
    n = len(row)
    out = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ri)
    return out


def _row_mask(row):
    """Bitmask of differences in a row."""
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
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _column_wise_construct(n_rows, num_elements, rng, max_scope):
    """
    Column-wise list scheduling: for each position (column), assign each row
    the smallest feasible value, with randomized tie-breaking among close
    candidates. Returns a list of rows or None if stuck.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0  # global mask of all used differences

    for pos in range(1, num_elements):
        # For each row, find the smallest feasible value at this position
        # Process rows in random order to break symmetry
        order = list(range(n_rows))
        rng.shuffle(order)

        # First pass: find the minimum feasible value for each row
        min_vals = {}
        for r_idx in order:
            row = rows[r_idx]
            prev_val = row[-1]
            found = False
            # Search from prev_val+1 upward
            for val in range(prev_val + 1, max_scope + 1):
                # Check within-row differences
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
                # Check against global used mask
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                min_vals[r_idx] = val
                found = True
                break
            if not found:
                return None

        # Second pass: assign values with some randomization among close candidates
        # Sort by min_val to process smaller values first (greedy)
        for r_idx in sorted(order, key=lambda i: min_vals[i]):
            row = rows[r_idx]
            prev_val = row[-1]
            min_val = min_vals[r_idx]
            
            # With some probability, try a slightly larger value for diversity
            if rng.random() < 0.3:
                search_start = min_val
                search_end = min(min_val + 5, max_scope)
            else:
                search_start = min_val
                search_end = min_val

            assigned = False
            for val in range(search_start, search_end + 1):
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
                row.append(val)
                used_mask |= new_mask
                assigned = True
                break
            
            if not assigned:
                # Try the minimum value (should always work)
                val = min_val
                new_diffs = set()
                for prev in row:
                    d = val - prev
                    new_diffs.add(d)
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    return None
                row.append(val)
                used_mask |= new_mask

    return rows


def _backtrack_row_with_mask(row_prefix, used_mask, num_elements, max_val):
    """
    Backtracking to complete a single row given a prefix.
    Returns the completed row or None.
    """
    if len(row_prefix) == num_elements:
        return row_prefix[:]
    
    row = row_prefix[:]
    last_val = row[-1]
    
    for val in range(last_val + 1, max_val + 1):
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
        
        result = _backtrack_row_with_mask(row + [val], used_mask | new_mask, num_elements, max_val)
        if result is not None:
            return result
    
    return None


def _precompute_row_catalog(n_rows, num_elements, max_scope, rng):
    """
    Precompute a catalog of valid rows (each row is a list of num_elements
    increasing integers starting at 0 with max value <= max_scope).
    Returns a list of (row, mask) pairs.
    """
    catalog = []
    used_global = 0
    
    for _ in range(n_rows):
        # Try to build a row that doesn't conflict with already-cataloged rows
        attempts = 0
        while attempts < 20:
            row = _backtrack_row_with_mask([0], used_global, num_elements, max_scope)
            if row is not None:
                mask = _row_mask(row)
                if mask & used_global:
                    attempts += 1
                    continue
                catalog.append((row, mask))
                used_global |= mask
                break
            attempts += 1
        else:
            return None
    
    return catalog


def _row_pair_swap_search(rows, num_elements, rng, time_limit, max_iter=50000):
    """
    Local search: swap entire rows between positions to escape plateaus.
    Also try small element adjustments within rows.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    
    # Precompute row masks
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    # Verify global mask consistency
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if row_masks[i] & row_masks[j]:
                # Conflict - shouldn't happen if rows are valid
                return best_rows, best_scope
    
    stagnation = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        # Randomly pick two rows to swap
        i, j = rng.sample(range(n_rows), 2)
        
        # Try swapping rows i and j
        old_i = current_rows[i][:]
        old_j = current_rows[j][:]
        
        current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
        
        # Check if still valid (masks don't change, so validity is preserved)
        new_scope = _scope(current_rows)
        
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            # Revert swap
            current_rows[i], current_rows[j] = old_i, old_j
            stagnation += 1
    
    return best_rows, best_scope


def _element_local_search(rows, num_elements, rng, time_limit, max_iter=100000):
    """
    Element-level local search: try to reduce individual elements to their
    minimum feasible value.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    # Precompute masks
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    stagnation = 0
    max_stagnation = 10000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            stagnation += 1
            continue
        
        # Try to find the minimum feasible value
        new_val = None
        for candidate in range(low, old_val):
            row[pos] = candidate
            # Check within-row
            diffs = set()
            valid = True
            for ii in range(n_elem):
                for jj in range(ii + 1, n_elem):
                    d = row[jj] - row[ii]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            # Check against other rows
            other_mask = global_mask & ~row_masks[row_idx]
            new_row_mask = 0
            for d in diffs:
                new_row_mask |= (1 << d)
            if new_row_mask & other_mask:
                continue
            new_val = candidate
            break
        
        if new_val is None:
            row[pos] = old_val
            stagnation += 1
            continue
        
        # Apply the move
        row[pos] = new_val
        old_row_mask = row_masks[row_idx]
        new_row_mask = 0
        for ii in range(n_elem):
            for jj in range(ii + 1, n_elem):
                d = row[jj] - row[ii]
                new_row_mask |= (1 << d)
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask
        
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            # Revert
            row[pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
            stagnation += 1
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Column-wise construction with multiple restarts
    # Try progressively tighter scopes
    for seed in range(20):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 31 + 17)
        
        # Start with a generous scope, tighten over restarts
        start_scope = min(200, counting_bound + 40 + seed * 2)
        end_scope = max(counting_bound - 5, start_scope - 15)
        
        for scope in range(start_scope, end_scope - 1, -1):
            rows = _column_wise_construct(n_rows, num_elements, rng_local, scope)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                break  # Found valid at this scope, try tighter scope next
    
    # Phase 2: Backtracking construction as backup
    if best_rows is None or best_scope > counting_bound + 15:
        for seed in range(10):
            if time.time() > DEADLINE - 60:
                break
            rng_local = random.Random(seed * 13 + 7)
            scope = min(180, counting_bound + 30)
            rows = []
            used_mask = 0
            success = True
            for _ in range(n_rows):
                row = _backtrack_row_with_mask([0], used_mask, num_elements, scope)
                if row is None:
                    success = False
                    break
                rows.append(row)
                used_mask |= _row_mask(row)
            if success and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # Fallback: sequential greedy
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
    
    # Phase 3: Row-pair swap search
    remaining = DEADLINE - time.time()
    if remaining > 50:
        improved_rows, improved_scope = _row_pair_swap_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 45,
            max_iter=50000
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Element local search
    if time.time() < DEADLINE - 30:
        improved_rows, improved_scope = _element_local_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20,
            max_iter=100000
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 5: Another column-wise restart with tighter scope if time permits
    if time.time() < DEADLINE - 40 and best_scope > counting_bound:
        for seed in range(5):
            if time.time() > DEADLINE - 25:
                break
            rng_local = random.Random(seed * 7 + 3)
            scope = best_scope - 1
            if scope < counting_bound:
                break
            rows = _column_wise_construct(n_rows, num_elements, rng_local, scope)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                break
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
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
