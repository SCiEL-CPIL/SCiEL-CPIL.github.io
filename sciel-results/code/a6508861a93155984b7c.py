import time
import random
import math
import sys


def _compute_row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _compute_row_mask(row):
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
        all_diffs.extend(_compute_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _backtrack_row(prefix, used_mask, num_elements, max_val, rng):
    """
    Complete a row given its prefix [0, ...] using backtracking.
    Returns completed row list or None.
    """
    if len(prefix) == num_elements:
        return prefix[:]
    
    row = prefix
    last_val = row[-1]
    
    # Generate candidate values in a randomized order, biased toward smaller values
    # to keep the scope tight
    candidates = list(range(last_val + 1, max_val + 1))
    
    # Shuffle with bias: smaller values are more likely to be tried first
    # Use a weighted shuffle: weight = 1/(1 + 0.1 * (val - last_val))
    if len(candidates) > 20:
        # Partial shuffle for performance: try a random subset first
        rng.shuffle(candidates)
        # Sort the first few by value to prefer smaller
        candidates_sorted = sorted(candidates[:min(15, len(candidates))])
        candidates_rest = candidates[min(15, len(candidates)):]
        candidates = candidates_sorted + candidates_rest
    else:
        rng.shuffle(candidates)
    
    for val in candidates:
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
        
        # Check against global used_mask
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        
        if new_mask & used_mask:
            continue
        
        # Recurse
        result = _backtrack_row(row + [val], used_mask | new_mask, num_elements, max_val, rng)
        if result is not None:
            return result
    
    return None


def _construct_rows_backtracking(n_rows, num_elements, rng, max_scope):
    """
    Construct n_rows rows with all elements <= max_scope using backtracking.
    Returns list of rows or None if failed.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = _backtrack_row([0], used_mask, num_elements, max_scope, rng)
        if row is None:
            return None
        rows.append(row)
        row_mask = _compute_row_mask(row)
        used_mask |= row_mask
    
    return rows


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=150):
    """
    Greedy construction with randomization for diversity.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            high = max_scope_hint
            
            # Generate candidates with randomization
            candidates = list(range(row[-1] + 1, high + 1))
            if len(candidates) > 30:
                # Weighted sampling
                weights = [1.0 / (1.0 + 0.3 * i) for i in range(len(candidates))]
                selected = []
                pool = list(range(len(candidates)))
                sample_size = min(20, len(candidates))
                for _ in range(sample_size):
                    total_w = sum(weights[i] for i in pool)
                    r = rng.random() * total_w
                    cum = 0
                    for idx, pi in enumerate(pool):
                        cum += weights[pi]
                        if r <= cum:
                            selected.append(candidates[pi])
                            pool.remove(pi)
                            break
                candidates = sorted(selected)
            
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
                
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            
            if not found:
                # Fallback: jump forward
                val = row[-1] + 10 + rng.randint(0, 20)
                row.append(val)
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                for d in new_diffs:
                    used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows


def _compute_all_diffs(rows):
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _compute_row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    row = rows[row_idx]
    n = len(row)
    
    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None
    
    old_val = row[pos]
    if new_val == old_val:
        return False, None, None
    
    row[pos] = new_val
    
    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)
    
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)
    
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None
    
    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val
    
    return True, new_global_mask, new_row_mask


def _local_search_reduce(rows, num_elements, rng, time_limit):
    """
    Local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 200000
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
            continue
        
        new_val = rng.randint(low, old_val - 1)
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if success:
            current_rows[row_idx][pos] = new_val
            old_row_mask = row_masks[row_idx]
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
    
    return best_rows, best_scope


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=30.0):
    """
    Simulated annealing to minimize scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    temp = initial_temp
    cooling = 0.9997
    min_temp = 1.0
    
    iterations = 0
    max_iter = 300000
    
    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 15
        
        if low >= high:
            continue
        
        if old_val > low + 2 and rng.random() < 0.6:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if not success:
            continue
        
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = new_global_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            current_rows[row_idx][pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
        
        temp *= cooling
    
    return best_rows, best_scope


def _row_swap_search(rows, num_elements, rng, time_limit, max_scope_target):
    """
    Row-level local search: try to replace individual rows with tighter alternatives
    that fit within the current global difference mask.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    
    for it in range(50000):
        if time.time() > time_limit:
            break
        
        # Pick a random row to try to improve
        row_idx = rng.randint(0, n_rows - 1)
        
        # Compute the mask of all other rows
        other_mask = 0
        for i in range(n_rows):
            if i != row_idx:
                other_mask |= _compute_row_mask(current_rows[i])
        
        # Try to construct a new row with max_val = current_scope (or lower)
        # using backtracking with the other rows' mask as forbidden
        target_scope = min(current_scope, max_scope_target)
        new_row = _backtrack_row([0], other_mask, num_elements, target_scope, rng)
        
        if new_row is not None:
            new_row_mask = _compute_row_mask(new_row)
            if new_row_mask & other_mask:
                continue
            
            # Check if this improves the scope
            current_scope_check = _scope(current_rows)
            new_scope = max(current_scope_check, new_row[-1])
            
            if new_scope < best_scope:
                current_rows[row_idx] = new_row
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
            elif new_row[-1] < current_rows[row_idx][-1]:
                # Even if overall scope doesn't improve, replace if this row is tighter
                # This helps with future improvements
                current_rows[row_idx] = new_row
    
    return best_rows, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin from 150s
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: CSP backtracking construction at various scopes
    # Try multiple seeds and scope targets
    scope_targets = []
    # Start from counting bound + small margin, go up
    for offset in range(0, 40, 5):
        scope_targets.append(counting_bound + offset)
    # Also try some higher scopes for diversity
    scope_targets.extend([counting_bound + 50, counting_bound + 70, counting_bound + 100])
    scope_targets = sorted(set(scope_targets))
    
    for seed in range(15):
        if time.time() > DEADLINE - 60:
            break
        rng_local = random.Random(seed * 17 + 3)
        
        # Try scopes from tightest to looser
        for scope in scope_targets:
            if time.time() > DEADLINE - 60:
                break
            rows = _construct_rows_backtracking(n_rows, num_elements, rng_local, scope)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                # If we found a solution near the counting bound, stop early
                if s <= counting_bound + 10:
                    break
        if best_scope <= counting_bound + 10:
            break
    
    # Phase 2: Greedy construction for diversity
    if best_rows is None or best_scope > counting_bound + 30:
        for seed in range(20):
            if time.time() > DEADLINE - 50:
                break
            rng_local = random.Random(seed * 11 + 5)
            max_hint = min(200, counting_bound + 60)
            rows = _greedy_construct(n_rows, num_elements, rng_local, max_hint)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # Fallback if nothing found
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
    
    # Phase 3: Local search to reduce scope
    remaining = DEADLINE - time.time()
    if remaining > 40:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 40
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Row swap search - try to replace rows with tighter ones
    remaining = DEADLINE - time.time()
    if remaining > 30:
        improved_rows, improved_scope = _row_swap_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 30,
            max_scope_target=best_scope
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 5: Simulated annealing for fine-tuning
    remaining = DEADLINE - time.time()
    if remaining > 30:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20,
            initial_temp=max(5.0, best_scope * 0.1)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 6: Final local search pass
    remaining = DEADLINE - time.time()
    if remaining > 20:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 10
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
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
