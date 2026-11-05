"""
Difference triangle set (7,5) solver.

Strategy:
1. Column-wise list-scheduling construction: build all rows simultaneously,
   processing positions left-to-right, choosing the earliest feasible value
   for each cell while respecting global difference constraints.
2. Row-pair swap search: try swapping entire rows between candidate solutions
   to find combinations with smaller scope.
3. Element-wise local search for final polish.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs_list(row):
    """Return list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _row_mask(row):
    """Return frozenset of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_diffs = []
    for r in rows:
        all_diffs.extend(_row_diffs_list(r))
    return len(all_diffs) == len(set(all_diffs))


def _compute_all_diffs(rows):
    """Return set of all differences used."""
    used = set()
    for r in rows:
        used |= _row_mask(r)
    return used


def _column_wise_construct(n_rows, k, deadline):
    """
    Column-wise list-scheduling construction.
    Build all rows simultaneously, processing positions 0..k.
    At each position, for each row, find the smallest value > previous
    that doesn't create duplicate differences globally.
    
    Returns list of n_rows rows, or None if stuck.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, k + 1):
        # For this position, find the earliest feasible value for each row
        # Process rows in order of their current last value (smallest first)
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            
            # Try values starting from prev_val + 1
            x = prev_val + 1
            while True:
                # Check if this value creates any duplicate differences
                new_diffs = set()
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # No conflicts, use this value
                    row.append(x)
                    used_diffs |= new_diffs
                    break
                x += 1
                # Safety: if we've gone too far, return None
                if x > 500:
                    return None
    
    return rows


def _column_wise_construct_randomized(n_rows, k, rng, deadline):
    """
    Randomized column-wise construction with multiple strategies.
    Returns best solution found.
    """
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple randomizations
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        for pos in range(1, k + 1):
            # Randomly shuffle row processing order
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            for idx in row_order:
                row = rows[idx]
                prev_val = row[-1]
                
                # Find candidates: try a range of values
                # Bias toward smaller values but allow some randomness
                x = prev_val + 1
                found = False
                max_try = prev_val + 50
                
                while x <= max_try:
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if not conflict:
                        # With some probability, skip this to explore larger values
                        if rng.random() < 0.7:
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                        # Otherwise, try next value
                    x += 1
                
                if not found:
                    # Try harder: go further
                    for x in range(prev_val + 1, prev_val + 200):
                        new_diffs = set()
                        conflict = False
                        for i in range(len(row)):
                            d = x - row[i]
                            if d in used_diffs:
                                conflict = True
                                break
                            new_diffs.add(d)
                        if not conflict:
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                
                if not found:
                    break
        
        if found and len(rows[0]) == k + 1:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
    
    return best_rows


def _row_pair_swap_search(rows, k, deadline):
    """
    Try swapping individual elements between rows to reduce scope.
    This is a more global search than element-wise local search.
    """
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    used = _compute_all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to reduce each element in each row
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                
                # Try smaller values
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    # Compute old and new differences for this position
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                            new_diffs.add(new_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                            new_diffs.add(row[i] - new_val)
                    
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        sc = _scope(rows)
                        if sc < best_scope:
                            best_scope = sc
                        improved = True
                        break
    
    return rows


def _try_move_element(rows, idx, pos, new_val, used):
    """
    Try moving element at rows[idx][pos] to new_val.
    Returns (success, new_used) if successful.
    """
    row = rows[idx]
    old_val = row[pos]
    
    old_diffs = set()
    new_diffs = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            old_diffs.add(old_val - row[i])
            new_diffs.add(new_val - row[i])
        else:
            old_diffs.add(row[i] - old_val)
            new_diffs.add(row[i] - new_val)
    
    remaining = used - old_diffs
    if new_diffs & remaining:
        return False, used
    
    new_used = (used - old_diffs) | new_diffs
    return True, new_used


def _element_local_search(rows, k, deadline):
    """
    Element-wise local search to reduce scope.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    success, new_used = _try_move_element(rows, idx, pos, new_val, used)
                    if success:
                        row[pos] = new_val
                        used = new_used
                        improved = True
                        break
    
    return rows


def _simulated_annealing(rows, k, deadline, rng):
    """
    Simulated annealing to escape local minima.
    Allows temporary scope increases to find better solutions.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline and T > 0.1:
        # Pick a random row and position
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        
        row = rows[idx]
        old_val = row[pos]
        
        # Pick a random new value
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure validity: must be > previous and < next
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
        if new_val <= 0:
            continue
        
        success, new_used = _try_move_element(rows, idx, pos, new_val, used)
        if not success:
            T *= cooling
            continue
        
        old_scope = _scope(rows)
        rows[idx][pos] = new_val
        used = new_used
        new_scope = _scope(rows)
        
        # Accept if better, or with probability exp(-delta/T)
        delta_scope = new_scope - old_scope
        if delta_scope <= 0 or rng.random() < math.exp(-delta_scope / T):
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            # Reject: revert
            rows[idx][pos] = old_val
            # Recompute used
            # Actually, we need to revert used too
            # This is tricky; let's just recompute
            used = _compute_all_diffs(rows)
        
        T *= cooling
    
    return best_rows


def _precompute_row_catalog(n_rows, k, max_scope=150):
    """
    Precompute a catalog of valid rows up to max_scope.
    Returns list of (row, mask) tuples.
    """
    catalog = []
    seen_masks = set()
    
    for scope in range(k, max_scope + 1):
        # Generate rows with max entry = scope
        # This is expensive; use greedy generation instead
        pass
    
    # Use greedy generation for diversity
    rng = random.Random(123)
    used_global = set()
    
    for _ in range(500):
        row = [0]
        new_diffs = set()
        x = 0
        stuck = False
        
        while len(row) < k + 1:
            x += 1
            if x > max_scope:
                stuck = True
                break
            new = {x - a for a in row}
            if new & new_diffs:
                continue
            row.append(x)
            new_diffs |= new
        
        if not stuck and len(row) == k + 1:
            mask = frozenset(new_diffs)
            if mask not in seen_masks:
                seen_masks.add(mask)
                catalog.append((row, mask))
    
    return catalog


def _construct(n_rows, k, deadline):
    """
    Main construction: column-wise + refinement.
    """
    start = time.time()
    
    # Step 1: Column-wise construction
    rows = _column_wise_construct(n_rows, k, deadline)
    
    if rows is None:
        # Fallback to pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_scope = _scope(rows)
    
    # Step 2: Randomized column-wise constructions
    if time.time() < deadline - 20:
        rng = random.Random(42)
        alt_rows = _column_wise_construct_randomized(n_rows, k, rng, deadline - 20)
        if alt_rows is not None:
            alt_scope = _scope(alt_rows)
            if alt_scope < best_scope:
                rows = alt_rows
                best_scope = alt_scope
    
    # Step 3: Row-pair swap search
    if time.time() < deadline - 15:
        rows = _row_pair_swap_search(rows, k, deadline - 15)
        best_scope = _scope(rows)
    
    # Step 4: Element local search
    if time.time() < deadline - 10:
        rows = _element_local_search(rows, k, deadline - 10)
        best_scope = _scope(rows)
    
    # Step 5: Simulated annealing for final polish
    if time.time() < deadline - 5:
        rng = random.Random(7)
        sa_rows = _simulated_annealing(rows, k, deadline - 5, rng)
        if _scope(sa_rows) < best_scope:
            rows = sa_rows
            best_scope = _scope(rows)
    
    # Final element local search
    if time.time() < deadline - 2:
        rows = _element_local_search(rows, k, deadline - 2)
    
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rows = _construct(n_rows, k, deadline)
    
    # Validate
    if rows is None or len(rows) != n_rows:
        # Ultimate fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
    
    return rows

# EVOLVE-BLOCK-END
