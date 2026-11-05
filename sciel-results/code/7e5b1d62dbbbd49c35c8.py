import time
import random
import math
import sys
import heapq
from itertools import combinations

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                all_diffs.append(row[j] - ri)
    return len(all_diffs) == len(set(all_diffs))

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    """
    Try to move element at rows[row_idx][pos] to new_val.
    Returns (success, new_global_mask, new_row_mask) or (False, None, None).
    """
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

def _row_major_backtrack(n_rows, num_elements, target_scope, rng, time_limit):
    """
    Row-major backtracking: build one complete row at a time.
    Uses bitmask for fast difference checking.
    Order rows by remaining feasibility (most constrained first).
    """
    start = time.time()
    rows = []
    used_mask = 0  # bitmask of all used differences
    
    # Precompute: for a row being built, track which diffs are used within it
    # We'll use recursive backtracking per row
    
    def build_row(current_row, used_mask_local, global_mask):
        """
        Recursively build a single row.
        current_row: list of values so far (starts with [0])
        used_mask_local: bitmask of diffs used within this row so far
        global_mask: bitmask of all diffs used in previously completed rows
        Returns: completed row or None if stuck
        """
        if len(current_row) == num_elements:
            return current_row[:]
        
        last_val = current_row[-1]
        # Try values from last_val+1 up to target_scope
        # Use a limited search window for efficiency
        max_search = target_scope
        found = False
        
        # Determine candidate range
        low = last_val + 1
        high = min(max_search, last_val + max(20, (max_search - last_val) // 4))
        
        # Generate candidates
        candidates = list(range(low, high + 1))
        # Add some random larger candidates for diversity
        if high < max_search:
            for _ in range(min(5, max_search - high)):
                candidates.append(rng.randint(high + 1, max_search))
        
        # Sort candidates: prefer smaller values (greedy) with some randomness
        if rng.random() < 0.5:
            candidates.sort()
        else:
            rng.shuffle(candidates)
        
        for val in candidates:
            if time.time() > time_limit:
                return None
            
            # Check if val creates valid differences
            local_new_mask = 0
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                bit = 1 << d
                if (used_mask_local & bit) or (global_mask & bit):
                    valid = False
                    break
                local_new_mask |= bit
            
            if valid:
                # Also check no duplicate within the new diffs themselves
                # (already handled by local_new_mask check)
                new_local_mask = used_mask_local | local_new_mask
                result = build_row(current_row + [val], new_local_mask, global_mask)
                if result is not None:
                    return result
        
        return None
    
    # Build rows one by one, ordering by tightness
    remaining_rows = list(range(n_rows))
    
    for _ in range(n_rows):
        if time.time() > time_limit:
            return None
        
        # Try to build a row for each remaining row index
        # Use the one that succeeds with smallest scope
        best_row = None
        best_scope_val = float('inf')
        
        for attempt in range(3):
            if time.time() > time_limit:
                break
            
            row = build_row([0], 0, used_mask)
            if row is not None:
                s = row[-1]
                if s < best_scope_val:
                    best_scope_val = s
                    best_row = row
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        used_mask |= _row_mask(best_row)
    
    return rows

def _local_search_refine(rows, num_elements, rng, time_limit):
    """
    Simple local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000
    
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

def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=50.0):
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
    cooling = 0.9995
    min_temp = 0.5
    
    iterations = 0
    max_iter = 200000
    
    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 10
        
        if low >= high:
            continue
        
        if old_val > low + 2 and rng.random() < 0.7:
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

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy:
    1. Row-major backtracking with bitmask (structural change from column-wise)
    2. Local search refinement via element moves
    3. Simulated annealing for further improvement
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Row-major backtracking with multiple seeds and scopes
    # Start with a generous scope and work down
    initial_scope = max(counting_bound + 30, 120)
    
    for seed in range(12):
        if time.time() > DEADLINE - 60:
            break
        
        rng_local = random.Random(seed * 17 + 3)
        
        # Try progressively tighter scopes
        for scope_offset in [0, 5, 10, 20]:
            target = initial_scope - scope_offset
            if target < counting_bound:
                continue
            
            time_limit = time.time() + 5.0
            rows = _row_major_backtrack(n_rows, num_elements, target, rng_local, time_limit)
            
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                break  # Found a solution at this scope, try next seed
    
    # Fallback: greedy construction if backtracking failed
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
    
    # Phase 2: Local search refinement
    if time.time() < DEADLINE - 30:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 35
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 3: Simulated annealing
    if time.time() < DEADLINE - 15:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 10,
            initial_temp=max(5.0, best_scope * 0.05)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
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
