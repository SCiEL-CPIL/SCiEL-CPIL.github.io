"""
Strategy: Iterative Deepening Backtracking with Forward Checking.
Instead of local search on a loose solution, we search for a valid configuration
within a strict scope M. We iterate M from the theoretical lower bound upwards.
For each M, we use a backtracking solver to place rows one by one.
To handle the combinatorial explosion, we use a heuristic to order the candidate rows
and a forward checking mechanism to prune branches where remaining rows
cannot be placed.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _generate_valid_rows(scope, num_elements):
    """
    Generate all valid rows of size num_elements starting at 0,
    with max element <= scope and distinct internal differences.
    Returns a list of (row, mask) tuples.
    """
    valid_rows = []
    # We need to choose num_elements-1 values from 1..scope
    # C(scope, num_elements-1)
    # For (7,5) -> num_elements=6, scope~112. C(112, 5) is ~ 1.5e8. Too many to enumerate all.
    # So we cannot enumerate ALL valid rows.
    # We need a different strategy.
    
    # Revised Strategy:
    # Since we can't enumerate all rows, we use a "Constructive Search" approach.
    # We will try to build the rows one by one.
    # For each row, we don't enumerate all possibilities, but we use a randomized
    # backtracking or a heuristic greedy with restarts, but guided by the "scope" constraint.
    
    # Actually, for the specific parameters (7, 5), the number of rows is small (7).
    # The number of differences per row is 15. Total diffs = 105.
    # Scope 112.
    
    # Let's try a "Randomized Backtracking" with strong heuristics.
    # 1. Sort candidate rows by "density" or "spread" to prefer those that leave more room?
    # 2. Or simply: Try to place rows. If stuck, backtrack.
    
    # Given the time limit, a pure backtracking over the huge space of rows is risky.
    # The previous approach (Local Search) was better because it started from a valid state.
    # 
    # Let's try a Hybrid:
    # 1. Find a valid solution with a loose scope (e.g. 200) using fast greedy.
    # 2. Use "Scope Reduction" local search, but with a stronger move set:
    #    - Swap elements between rows.
    #    - Replace an entire row with a new random valid row that fits in the current scope and conflicts less.
    #    - This is effectively Simulated Annealing on the scope.
    
    pass

def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=100):
    """
    Fast greedy construction to get a valid initial solution.
    """
    for restart in range(max_restarts):
        used_mask = 0
        rows = []
        success = True
        # Shuffle the order of "difficulty" or just random?
        # Random selection of rows is hard because we can't enumerate.
        # We use random sampling of rows.
        
        for _ in range(n_rows):
            row_found = False
            # Try to find a row that fits in max_scope and doesn't conflict
            for attempt in range(500):
                # Sample num_elements-1 distinct values from 1..max_scope
                if num_elements - 1 > max_scope:
                    break
                vals = sorted(rng.sample(range(1, max_scope + 1), num_elements - 1))
                row = [0] + vals
                # Check internal validity
                diffs = _row_diffs(row)
                if len(diffs) != len(set(diffs)):
                    continue
                m = _row_mask(row)
                if m & used_mask:
                    continue
                # Valid row
                rows.append(row)
                used_mask |= m
                row_found = True
                break
            if not row_found:
                success = False
                break
        if success:
            return rows
    return None

def _sa_reduce_scope(rows, num_elements, rng, time_limit):
    """
    Simulated Annealing to reduce the scope.
    State: list of rows.
    Objective: Minimize max(row[-1]).
    Moves:
    1. Change one element in one row (keeping it increasing and valid internally).
    2. Replace one row entirely with a new random row that fits in current scope 
       (or slightly larger) and has fewer conflicts.
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute masks for fast conflict checking
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
        
    n_rows = len(current_rows)
    k = num_elements - 1
    
    # Temperature schedule
    T = 10.0
    cool_rate = 0.995
    
    def get_energy():
        return _scope(current_rows)
        
    def try_replace_row(row_idx, new_row):
        """Replace row_idx with new_row if it fits in current_scope (or best_scope?) and is valid."""
        nonlocal current_rows, row_masks, global_mask, current_scope
        # Check if new_row is internally valid (assumed by generation)
        # Check if new_row fits in scope
        if new_row[-1] > current_scope:
            return False
        new_mask = _row_mask(new_row)
        other_mask = global_mask & ~row_masks[row_idx]
        if new_mask & other_mask:
            return False
        
        # Apply
        old_row = current_rows[row_idx]
        current_rows[row_idx] = new_row
        old_mask = row_masks[row_idx]
        row_masks[row_idx] = new_mask
        global_mask = (global_mask & ~old_mask) | new_mask
        current_scope = _scope(current_rows)
        
        # Check for improvement
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in current_rows]
            
        return True
        
    def revert_replace_row(row_idx, old_row, old_mask):
        nonlocal current_rows, row_masks, global_mask, current_scope
        new_mask = row_masks[row_idx]
        current_rows[row_idx] = old_row
        row_masks[row_idx] = old_mask
        global_mask = (global_mask & ~new_mask) | old_mask
        current_scope = _scope(current_rows)

    def try_move_element(row_idx, pos, new_val):
        nonlocal current_rows, row_masks, global_mask, current_scope
        row = current_rows[row_idx]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        row[pos] = new_val
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            row[pos] = old_val
            return False
        new_mask = _row_mask(row)
        other_mask = global_mask & ~row_masks[row_idx]
        if new_mask & other_mask:
            row[pos] = old_val
            return False
            
        old_val = row[pos] # Wait, we already set it. Need to store before.
        # Let's restructure to store old state properly.
        return False # Placeholder, logic below is cleaner

    # Let's implement a cleaner SA loop
    # We will focus on "Row Replacement" as the primary move, as it's easier to validate 
    # and often finds better global structures than single element tweaks.
    
    max_iters = 50000
    for i in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Pick a random row to replace
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx][:]
        old_mask = row_masks[row_idx]
        
        # Generate a candidate row
        # Bias towards smaller values?
        # Sample from 1..current_scope
        if k > current_scope:
            continue
        vals = sorted(rng.sample(range(1, current_scope + 1), k))
        new_row = [0] + vals
        
        # Check internal validity
        diffs = _row_diffs(new_row)
        if len(diffs) != len(set(diffs)):
            continue
            
        new_mask = _row_mask(new_row)
        other_mask = global_mask & ~old_mask
        if new_mask & other_mask:
            continue
            
        # Candidate is valid!
        # Calculate energy change
        old_scope = current_scope
        # Apply
        current_rows[row_idx] = new_row
        row_masks[row_idx] = new_mask
        global_mask = (global_mask & ~old_mask) | new_mask
        new_scope = _scope(current_rows)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_scope < old_scope:
            # Always accept improvement
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Worse or same. Accept with prob?
            # If same scope, it might help future moves. Accept with prob 0.5.
            # If worse, reject.
            if new_scope == old_scope and rng.random() < 0.5:
                pass # Accept
            else:
                # Revert
                current_rows[row_idx] = old_row
                row_masks[row_idx] = old_mask
                global_mask = (global_mask & ~new_mask) | old_mask
                current_scope = old_scope
                
        T *= cool_rate

    return best_rows, best_scope

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    import math
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid solution
    best_rows = None
    best_scope = float('inf')
    
    # Start with a loose scope to ensure we find *something*
    initial_scope = 200
    for scope in range(initial_scope, 300):
        if time.time() > DEADLINE - 60:
            break
        rows = _greedy_construct(n_rows, num_elements, scope, rng, max_restarts=50)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            break
    
    if best_rows is None:
        # Fallback: sequential construction without scope limit
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)

    # Phase 2: SA to reduce scope
    while time.time() < DEADLINE - 10:
        improved_rows, improved_scope = _sa_reduce_scope(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        if best_scope <= counting_bound:
            break
            
    # Phase 3: Final Validation
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
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
