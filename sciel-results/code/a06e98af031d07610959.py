import time
import random
import math
import sys


def _row_diffs(row):
    """Return set of all positive differences within a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


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
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _count_available_values(row_prefix, used_mask, max_val, num_elements):
    """
    For each position in the row (from len(row_prefix) to num_elements-1),
    count how many values are available (satisfying increasing order and
    not conflicting with used_mask or within-row diffs).
    Returns the minimum count across all remaining positions, or -1 if any
    position has 0 available values.
    """
    row = row_prefix
    last_val = row[-1]
    min_count = float('inf')
    
    for pos in range(len(row), num_elements):
        # The next value must be > last_val and <= max_val
        # Also, for pos > len(row), the lower bound is determined by previous choices
        # But for a quick feasibility check, we just need SOME value in range
        low = last_val + 1
        high = max_val
        
        if low > high:
            return -1
        
        count = 0
        # We need to check if there's at least one valid value
        # A value v is valid if:
        # 1. v > last_val
        # 2. v <= max_val  
        # 3. For all existing elements a in row: (v - a) not in used_mask and (v - a) not in row_diffs
        # 4. All (v - a) for a in row are distinct
        
        # For speed, just check a sample or do a quick scan
        # For the feasibility check, let's just verify there's at least one
        found_any = False
        for v in range(low, high + 1):
            new_diffs = set()
            valid = True
            for a in row:
                d = v - a
                if d in new_diffs:
                    valid = False
                    break
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                found_any = True
                break
        
        if not found_any:
            return -1
    
    return 1  # At least one value exists for each position (conservative)


def _backtrack_row(row_prefix, used_mask, num_elements, max_val, rng, time_check=None, deadline=None):
    """
    Backtracking to complete a row given a prefix.
    Uses randomized ordering for exploration.
    Returns the completed row or None if stuck.
    """
    if time_check is not None and time_check() > deadline:
        return None
    
    if len(row_prefix) == num_elements:
        return row_prefix
    
    row = row_prefix[:]
    last_val = row[-1]
    
    # Collect valid candidates
    candidates = []
    for val in range(last_val + 1, max_val + 1):
        new_diffs = set()
        valid = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                valid = False
                break
            if (used_mask >> d) & 1:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            candidates.append(val)
    
    if not candidates:
        return None
    
    # Randomize candidate order for diversity
    rng.shuffle(candidates)
    
    # Try each candidate
    for val in candidates:
        if time_check is not None and time_check() > deadline:
            return None
        
        new_diffs = set()
        for prev in row:
            new_diffs.add(val - prev)
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        
        result = _backtrack_row(row + [val], used_mask | new_mask, num_elements, max_val, rng, time_check, deadline)
        if result is not None:
            return result
    
    return None


def _construct_rows_backtracking(n_rows, num_elements, max_val, rng, time_check=None, deadline=None):
    """
    Try to construct all rows with maximum value <= max_val using backtracking.
    Returns list of rows or None if failed.
    """
    if time_check is not None and time_check() > deadline:
        return None
    
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        if time_check is not None and time_check() > deadline:
            return None
        
        row = [0]
        result = _backtrack_row(row, used_mask, num_elements, max_val, rng, time_check, deadline)
        
        if result is None:
            return None
        
        rows.append(result)
        row_mask = _row_mask(result)
        used_mask |= row_mask
    
    return rows


def _greedy_fallback(n_rows, num_elements):
    """Simple sequential greedy construction as a fallback."""
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _local_search_reduce(rows, num_elements, rng, time_limit):
    """
    Local search to reduce scope by moving elements down.
    Uses bitmask for fast collision checking.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask = 0
    row_masks = []
    for row in current_rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    stagnation = 0
    max_stagnation = 3000
    
    while time.time() < time_limit and stagnation < max_stagnation:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            stagnation += 1
            continue
        
        # Try to move to a lower value
        new_val = rng.randint(low, old_val - 1)
        
        # Check validity
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        n = n_elem
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
            if not valid:
                break
        
        if valid:
            new_row_mask = 0
            for d in diffs_in_row:
                new_row_mask |= (1 << d)
            
            other_mask = global_mask & ~row_masks[row_idx]
            if new_row_mask & other_mask:
                valid = False
        
        if not valid:
            row[pos] = old_val
            stagnation += 1
            continue
        
        # Accept the move
        row[pos] = new_val
        old_row_mask = row_masks[row_idx]
        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask
        
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            stagnation += 1
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Backtracking construction with progressively tighter scope bounds
    # Start from a generous bound and tighten
    # The counting bound is a lower bound; we search around it
    
    # Try multiple seeds with different scope targets
    scope_attempts = []
    # Include the counting bound and slightly above
    base = max(counting_bound, 10)
    for offset in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]:
        scope_attempts.append(base + offset)
    # Also try some higher values
    for offset in [60, 80, 100]:
        scope_attempts.append(base + offset)
    
    # Sort in increasing order to find the tightest first
    scope_attempts.sort()
    
    # Deduplicate
    seen = set()
    unique_attempts = []
    for s in scope_attempts:
        if s not in seen:
            seen.add(s)
            unique_attempts.append(s)
    scope_attempts = unique_attempts
    
    # Try backtracking with each scope, using different seeds
    for attempt_idx, target_scope in enumerate(scope_attempts):
        if time.time() > DEADLINE - 60:
            break
        
        # Use multiple seeds for each scope
        for seed in range(3):
            if time.time() > DEADLINE - 60:
                break
            
            rng_local = random.Random(seed * 1000 + attempt_idx * 7 + 13)
            
            # Add some randomization to the target scope
            actual_scope = target_scope + rng_local.randint(0, 3)
            
            rows = _construct_rows_backtracking(
                n_rows, num_elements, actual_scope, rng_local,
                time_check=time.time, deadline=DEADLINE - 60
            )
            
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                
                # If we found a solution at the counting bound or close, stop early
                if s <= counting_bound + 2:
                    break
        
        if best_rows is not None and best_scope <= counting_bound + 5:
            break
    
    # Phase 2: If backtracking didn't find a good solution, use greedy fallback
    if best_rows is None or best_scope > counting_bound + 20:
        for seed in range(5):
            if time.time() > DEADLINE - 40:
                break
            rng_local = random.Random(seed * 31 + 17)
            
            # Randomized greedy: build rows one by one with randomization
            rows = []
            used_mask = 0
            success = True
            
            for row_idx in range(n_rows):
                row = [0]
                for pos in range(1, num_elements):
                    found = False
                    # Try values in randomized order
                    candidates = list(range(row[-1] + 1, row[-1] + 50))
                    rng_local.shuffle(candidates)
                    
                    for val in candidates:
                        new_diffs = set()
                        valid = True
                        for prev in row:
                            d = val - prev
                            if d in new_diffs:
                                valid = False
                                break
                            if (used_mask >> d) & 1:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            row.append(val)
                            for d in new_diffs:
                                used_mask |= (1 << d)
                            found = True
                            break
                    
                    if not found:
                        # Fallback: find the smallest valid value
                        val = row[-1] + 1
                        while True:
                            new_diffs = set()
                            valid = True
                            for prev in row:
                                d = val - prev
                                if d in new_diffs or (used_mask >> d) & 1:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                row.append(val)
                                for d in new_diffs:
                                    used_mask |= (1 << d)
                                break
                            val += 1
                
                rows.append(row)
            
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # If no valid solution found, use sequential fallback
    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, num_elements)
        best_scope = _scope(best_rows)
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 25:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Another round of backtracking with the current best scope as target
    if time.time() < DEADLINE - 40 and best_scope > counting_bound:
        # Try to find a solution with scope <= best_scope - 1
        target = best_scope - 1
        if target >= counting_bound:
            for seed in range(5):
                if time.time() > DEADLINE - 40:
                    break
                rng_local = random.Random(seed * 23 + 5)
                rows = _construct_rows_backtracking(
                    n_rows, num_elements, target, rng_local,
                    time_check=time.time, deadline=DEADLINE - 40
                )
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
        return _greedy_fallback(n_rows, num_elements)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
