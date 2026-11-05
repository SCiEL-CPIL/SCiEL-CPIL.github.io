import time
import random
import math
import numpy as np


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
    seen = set()
    for row in rows:
        for d in _row_diffs_list(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _flat_to_rows(flat, n_rows, num_elements):
    """Convert flat array to list of rows."""
    rows = []
    for i in range(n_rows):
        rows.append(flat[i * num_elements:(i + 1) * num_elements].tolist())
    return rows


def _rows_to_flat(rows, n_rows, num_elements):
    """Convert list of rows to flat numpy array."""
    flat = np.zeros(n_rows * num_elements, dtype=np.int32)
    for i, row in enumerate(rows):
        flat[i * num_elements:(i + 1) * num_elements] = row
    return flat


def _compute_conflicts(flat, n_rows, num_elements, max_scope):
    """
    Vectorized conflict computation.
    Returns (conflict_count, scope).
    conflict_count: number of duplicated differences (0 means valid).
    """
    # Build all differences for all rows
    # For each row, compute all pairwise differences
    # flat is shape (n_rows, num_elements)
    rows = flat.reshape(n_rows, num_elements)
    
    # All pairwise differences: for each row, (j - i) pairs
    # num_pairs per row
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs = n_rows * num_pairs
    
    # Compute all differences
    diffs = np.zeros(total_diffs, dtype=np.int32)
    idx = 0
    for r in range(n_rows):
        row = rows[r]
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                diffs[idx] = row[j] - row[i]
                idx += 1
    
    # Count duplicates: sort and count adjacent equal
    if total_diffs == 0:
        return 0, 0
    
    sorted_diffs = np.sort(diffs)
    # Count how many elements are duplicates (i.e., appear more than once)
    # A value v appearing c times contributes (c-1) to conflict count
    changes = np.where(np.diff(sorted_diffs) != 0)[0]
    # Number of unique values
    n_unique = len(changes) + 1
    conflict_count = total_diffs - n_unique
    
    scope = int(flat[-1]) if n_rows > 0 else 0
    return conflict_count, scope


def _sa_optimize(flat, n_rows, num_elements, rng, deadline, initial_temp=50.0):
    """
    Simulated annealing on flat array.
    Move: pick a random row, pick a random position (1..num_elements-1),
    change value by a small random delta.
    Cost: conflict_count + scope * 0.1
    """
    cur_flat = flat.copy()
    best_flat = flat.copy()
    
    # Initial cost
    cur_conf, cur_scope = _compute_conflicts(cur_flat, n_rows, num_elements, 0)
    cur_cost = cur_conf + cur_scope * 0.1
    best_cost = cur_cost
    best_conf, best_scope = cur_conf, cur_scope
    
    temp = initial_temp
    cooling = 0.99995
    min_temp = 0.01
    n_elem = num_elements
    
    while time.time() < deadline and temp > min_temp:
        # Pick move
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        offset = row_idx * n_elem + pos
        old_val = int(cur_flat[offset])
        
        # Delta: small random change
        delta = rng.randint(-5, 5)
        if delta == 0:
            temp *= cooling
            continue
        new_val = old_val + delta
        
        # Bounds: must be > prev and < next (or within scope)
        prev_val = int(cur_flat[offset - 1]) if pos > 0 else -1
        next_val = int(cur_flat[offset + 1]) if pos < n_elem - 1 else 200
        
        if new_val <= prev_val or new_val >= next_val:
            temp *= cooling
            continue
        
        # Apply move
        cur_flat[offset] = new_val
        
        # Compute new cost
        new_conf, new_scope = _compute_conflicts(cur_flat, n_rows, num_elements, 0)
        new_cost = new_conf + new_scope * 0.1
        
        delta_cost = new_cost - cur_cost
        if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / temp):
            cur_cost = new_cost
            cur_conf, cur_scope = new_conf, new_scope
            if new_conf == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_conf = 0
                best_flat = cur_flat.copy()
                best_cost = new_cost
        else:
            cur_flat[offset] = old_val
        
        temp *= cooling
    
    return best_flat, best_scope, best_conf


def _greedy_initial(n_rows, num_elements, rng):
    """Greedy construction: build rows one by one, smallest valid value per position."""
    flat = np.zeros(n_rows * num_elements, dtype=np.int32)
    used = set()
    
    for r in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            for val in range(row[-1] + 1, 300):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                if new_diffs & used:
                    continue
                row.append(val)
                used |= new_diffs
                found = True
                break
            if not found:
                # Fallback
                val = row[-1] + 20 + rng.randint(0, 30)
                row.append(val)
                for d in (val - prev for prev in row[:-1]):
                    used.add(d)
        flat[r * num_elements:(r + 1) * num_elements] = row
    
    return flat


def _local_search_reduce(flat, n_rows, num_elements, rng, time_limit):
    """
    Fast local search: try to reduce scope by moving elements downward.
    Only accept if conflict stays 0.
    """
    start = time.time()
    cur_flat = flat.copy()
    best_flat = flat.copy()
    
    cur_conf, cur_scope = _compute_conflicts(cur_flat, n_rows, num_elements, 0)
    best_scope = cur_scope
    n_elem = num_elements
    stagnation = 0
    max_stag = 50000
    
    while time.time() < time_limit and stagnation < max_stag:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        offset = row_idx * n_elem + pos
        old = int(cur_flat[offset])
        low = int(cur_flat[offset - 1]) + 1
        high = int(cur_flat[offset + 1]) - 1 if pos < n_elem - 1 else cur_scope
        if low >= old:
            stagnation += 1
            continue
        new_val = rng.randint(low, old - 1)
        if new_val == old:
            stagnation += 1
            continue
        
        cur_flat[offset] = new_val
        new_conf, new_scope = _compute_conflicts(cur_flat, n_rows, num_elements, 0)
        
        if new_conf == 0:
            if new_scope < best_scope:
                best_scope = new_scope
                best_flat = cur_flat.copy()
                stagnation = 0
            else:
                cur_flat[offset] = old
                stagnation += 1
        else:
            cur_flat[offset] = old
            stagnation += 1
    
    return best_flat, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_flat = None
    best_scope = float('inf')
    
    # Phase 1: Greedy initial solutions with multiple seeds
    for seed in range(10):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 13 + 7)
        flat = _greedy_initial(n_rows, num_elements, rng_local)
        conf, scope = _compute_conflicts(flat, n_rows, num_elements, 0)
        if conf == 0 and scope < best_scope:
            best_scope = scope
            best_flat = flat.copy()
    
    # Fallback if no valid solution found
    if best_flat is None:
        # Sequential fallback
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
        best_flat = _rows_to_flat(rows, n_rows, num_elements)
        best_scope = _scope(rows)
    
    # Phase 2: SA optimization
    if time.time() < DEADLINE - 40:
        for seed in range(4):
            if time.time() > DEADLINE - 35:
                break
            rng_local = random.Random(seed * 23 + 11)
            # Start from best or a perturbed version
            start_flat = best_flat.copy()
            # Perturb
            for _ in range(rng_local.randint(1, 5)):
                ri = rng_local.randint(0, n_rows - 1)
                pi = rng_local.randint(1, num_elements - 1)
                off = ri * num_elements + pi
                lo = int(start_flat[off - 1]) + 1
                hi = int(start_flat[off + 1]) - 1 if pi < num_elements - 1 else best_scope + 10
                if lo < hi:
                    start_flat[off] = rng_local.randint(lo, hi)
            
            sa_flat, sa_scope, sa_conf = _sa_optimize(
                start_flat, n_rows, num_elements, rng_local,
                DEADLINE - 30, initial_temp=max(10.0, best_scope * 0.2))
            
            if sa_conf == 0 and sa_scope < best_scope:
                best_scope = sa_scope
                best_flat = sa_flat
    
    # Phase 3: Local search to reduce scope
    if time.time() < DEADLINE - 15:
        for seed in range(3):
            if time.time() > DEADLINE - 12:
                break
            rng_local = random.Random(seed * 31 + 17)
            ls_flat, ls_scope = _local_search_reduce(
                best_flat, n_rows, num_elements, rng_local, DEADLINE - 10)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_flat = ls_flat
    
    # Phase 4: Final SA polish
    if time.time() < DEADLINE - 5 and best_scope > counting_bound:
        rng_local = random.Random(999)
        sa_flat, sa_scope, sa_conf = _sa_optimize(
            best_flat, n_rows, num_elements, rng_local,
            DEADLINE - 3, initial_temp=max(5.0, best_scope * 0.1))
        if sa_conf == 0 and sa_scope < best_scope:
            best_scope = sa_scope
            best_flat = sa_flat
    
    # Convert to rows
    rows = _flat_to_rows(best_flat, n_rows, num_elements)
    
    # Final validation
    if _is_valid(rows):
        return rows
    else:
        # Fallback
        rows2 = []
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
            rows2.append(row)
        return rows2


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
