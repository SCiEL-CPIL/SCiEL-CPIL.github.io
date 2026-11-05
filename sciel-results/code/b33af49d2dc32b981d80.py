import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    """
    Returns a bitmask of differences for a row.
    Assumes differences fit in an integer (they do, M < 1000).
    """
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            m |= (1 << d)
    return m

def _is_row_valid_internal(row):
    """
    Checks if a single row has distinct differences.
    """
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _get_max_scope(rows):
    return max(r[-1] for r in rows)

def _construct_valid_row(k, max_val, used_mask, rng, max_tries=1000):
    """
    Constructs a single valid row of length k+1 (k=5 -> 6 elements) 
    with max <= max_val, no internal diff conflicts, and no external conflict with used_mask.
    """
    num_elements = k + 1
    
    for _ in range(max_tries):
        # Generate 5 random distinct integers from 1 to max_val
        if max_val < 5:
            return None
        vals = rng.sample(range(1, max_val + 1), num_elements - 1)
        vals.sort()
        row = [0] + vals
        
        # Check internal validity
        if not _is_row_valid_internal(row):
            continue
            
        # Check external validity
        m = _row_mask(row)
        if m & used_mask:
            continue
            
        return row
        
    return None

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Leave 5s for cleanup/return
    
    rng = random.Random(12345)
    
    # k=5 means 6 elements per row.
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Initial scope estimate. Counting bound is total_diffs_needed.
    # Best known is 112. Let's start higher to ensure we find a valid init.
    initial_scope = 150
    
    # Phase 1: Initialize a valid population
    # We just need one valid solution to start the annealing.
    # We can also keep a small population for diversity.
    
    def generate_initial_solution(scope_limit):
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = _construct_valid_row(k, scope_limit, used_mask, rng, max_tries=2000)
            if row is None:
                return None
            rows.append(row)
            used_mask |= _row_mask(row)
        return rows
        
    # Try to get a valid solution
    current_rows = None
    # Try tighter scopes first to get a good start? 
    # Or just start wide and narrow down.
    for attempt_scope in [120, 130, 140, 150, 200]:
        current_rows = generate_initial_solution(attempt_scope)
        if current_rows:
            break
            
    if not current_rows:
        # Fallback: Greedy construction without strict scope limit, just grow as needed
        # This guarantees a solution, though potentially with large scope.
        used_mask = 0
        current_rows = []
        for i in range(n_rows):
            scope_guess = 150 + i * 10
            row = None
            while row is None:
                row = _construct_valid_row(k, scope_guess, used_mask, rng, max_tries=100)
                if row is None:
                    scope_guess += 5
            current_rows.append(row)
            used_mask |= _row_mask(row)
            
    best_rows = [r[:] for r in current_rows]
    best_scope = _get_max_scope(best_rows)
    
    # Phase 2: Simulated Annealing / Local Search
    # State: list of rows
    # We maintain the global mask for O(1) conflict checks.
    
    def compute_global_mask(rows):
        gm = 0
        for r in rows:
            gm |= _row_mask(r)
        return gm
        
    def check_full_validity(rows):
        """
        Returns True if all diffs are distinct globally and internally.
        """
        gm = 0
        for r in rows:
            # Internal check
            if not _is_row_valid_internal(r):
                return False
            m = _row_mask(r)
            if m & gm:
                return False
            gm |= m
        return True
        
    # Main SA Loop
    current_rows = best_rows
    current_scope = best_scope
    current_global_mask = compute_global_mask(current_rows)
    
    # Calculate masks for each row to avoid recomputation
    row_masks = [_row_mask(r) for r in current_rows]
    
    T = 10.0
    cooling_rate = 0.9995
    restart_count = 0
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() < DEADLINE - 5:
        if T < 0.05:
            # Restart with a random kick or higher T if stuck
            if restart_count < 5:
                # Perturb current best
                current_rows = best_rows[:]
                current_scope = best_scope
                current_global_mask = compute_global_mask(current_rows)
                row_masks = [_row_mask(r) for r in current_rows]
                T = 5.0
                restart_count += 1
            else:
                break
        
        no_improve_steps += 1
        if no_improve_steps > max_no_improve:
            T = max(T, 5.0)
            no_improve_steps = 0
        
        # Pick a random row to modify
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        old_mask = row_masks[r_idx]
        
        # Generate a candidate move
        # Types of moves:
        # 1. Shift one element by +/- 1
        # 2. Swap two elements (if order preserved)
        # 3. Replace one element with a random valid one
        
        move_type = rng.random()
        candidate_row = row[:]
        
        if move_type < 0.6:
            # Shift
            pos = rng.randint(1, len(row) - 1) # Don't shift 0 (index 0)
            delta = rng.choice([-1, 1])
            new_val = row[pos] + delta
            
            # Check bounds: must be > prev and < next (if exists)
            if pos > 0 and new_val <= row[pos-1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos+1]:
                continue
            if new_val < 0:
                continue
                
            candidate_row[pos] = new_val
            
        elif move_type < 0.8:
            # Swap
            if len(row) > 2:
                p1, p2 = rng.sample(range(1, len(row)), 2)
                if p1 > p2: p1, p2 = p2, p1
                # Try swapping values at p1 and p2
                # We must ensure strict increase
                # Let's just swap and check
                candidate_row[p1] = row[p2]
                candidate_row[p2] = row[p1]
                if not all(candidate_row[i] < candidate_row[i+1] for i in range(len(candidate_row)-1)):
                    continue
            else:
                continue
                
        else:
            # Replace
            pos = rng.randint(1, len(row) - 1)
            low = row[pos-1] + 1
            high = row[pos+1] - 1 if pos < len(row) - 1 else current_scope + 10
            if low > high:
                continue
            # Bias towards smaller values to reduce scope
            if rng.random() < 0.7:
                high = min(high, row[pos])
            if low > high:
                continue
            new_val = rng.randint(low, high)
            if new_val == row[pos]:
                continue
            candidate_row[pos] = new_val
        
        # Validate internal differences of candidate_row
        if not _is_row_valid_internal(candidate_row):
            continue
            
        # Calculate new mask
        new_mask = _row_mask(candidate_row)
        
        # Check global conflict
        # The global mask without the current row is: current_global_mask ^ old_mask
        # But bitwise XOR only works if old_mask is a subset of current_global_mask (which it is).
        # However, if there are overlaps in differences *within* the old row? No, internal diffs are unique.
        # So old_mask is disjoint from other rows' masks.
        # Global mask = OR of all row masks.
        # Rest mask = current_global_mask & ~old_mask
        rest_mask = current_global_mask & ~old_mask
        
        if new_mask & rest_mask:
            continue
            
        # Move is valid!
        new_scope = max(candidate_row[-1], _get_max_scope([r for i, r in enumerate(current_rows) if i != r_idx]))
        
        # Note: _get_max_scope is O(n). We can optimize by keeping a max-heap or just computing.
        # For n=7, it's fast enough.
        
        delta_scope = new_scope - current_scope
        
        accept = False
        if delta_scope < 0:
            accept = True
        else:
            if rng.random() < math.exp(-delta_scope / T):
                accept = True
                
        if accept:
            # Update state
            current_rows[r_idx] = candidate_row
            current_scope = new_scope
            row_masks[r_idx] = new_mask
            # Update global mask: remove old, add new
            current_global_mask = (current_global_mask & ~old_mask) | new_mask
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                no_improve_steps = 0
                
        T *= cooling_rate
        
    # Final validation
    if check_full_validity(best_rows):
        return best_rows
    else:
        # Emergency fallback
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
