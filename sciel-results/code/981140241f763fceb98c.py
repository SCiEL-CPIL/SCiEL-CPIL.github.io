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
            if not (new_diffs & used) and len(new_diffs) == len(new_diffs):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _csp_construct(n_rows, num_elements, max_scope, rng, deadline, time_check):
    """
    CSP-based construction using forward checking and backjumping.
    Variables are (row_idx, pos) pairs, ordered by MRV (most constrained first).
    Domains are values from (prev_val+1) to max_scope.
    """
    if time_check() > deadline:
        return None
    
    total_vars = n_rows * num_elements
    # Variable ordering: (row_idx, pos)
    # We'll use a priority queue based on domain size (MRV)
    
    # Initialize: all rows start with [0]
    rows = [[0] + [None] * (num_elements - 1) for _ in range(n_rows)]
    used_mask = 0
    
    # Track which positions are assigned
    assigned = [[False] * num_elements for _ in range(n_rows)]
    for r in range(n_rows):
        assigned[r][0] = True
    
    # Compute initial domains and constraint propagation
    # For each unassigned position, compute valid values
    
    def get_domain(r, pos):
        """Get valid values for position (r, pos)."""
        if assigned[r][pos]:
            return [rows[r][pos]]
        
        # Lower bound: must be > previous value in same row
        if pos > 0:
            if rows[r][pos - 1] is None:
                return []  # Previous not assigned, can't determine
            low = rows[r][pos - 1] + 1
        else:
            low = 1  # pos 0 is always 0
        
        high = max_scope
        
        if low > high:
            return []
        
        # Check constraints: for each value, verify no conflict
        valid_vals = []
        for v in range(low, high + 1):
            # Check against all assigned positions in this row
            conflict = False
            for p in range(num_elements):
                if assigned[r][p] and p != pos:
                    d = abs(v - rows[r][p])
                    if d == 0:
                        conflict = True
                        break
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
            if not conflict:
                valid_vals.append(v)
        
        return valid_vals
    
    # Use recursive backtracking with MRV
    # We'll track the state and use a stack
    
    # For efficiency, we'll use a simpler approach: 
    # Process rows in order, but within each row use MRV for positions
    
    def backtrack(row_idx, pos):
        """Backtrack to assign position (row_idx, pos) and all subsequent."""
        if time_check() > deadline:
            return False
        
        # Find next unassigned position
        next_r = row_idx
        next_pos = pos
        found = False
        for r in range(row_idx, n_rows):
            for p in range(num_elements):
                if r > row_idx or (r == row_idx and p >= pos):
                    if not assigned[r][p]:
                        next_r = r
                        next_pos = p
                        found = True
                        break
            if found:
                break
        
        if not found:
            return True  # All assigned
        
        # Get domain for this position
        domain = get_domain(next_r, next_pos)
        if not domain:
            return False
        
        # Shuffle domain for diversity
        rng.shuffle(domain)
        
        for v in domain:
            if time_check() > deadline:
                return False
            
            # Assign
            rows[next_r][next_pos] = v
            assigned[next_r][next_pos] = True
            
            # Update used_mask
            new_diffs = set()
            for p in range(num_elements):
                if assigned[next_r][p] and p != next_pos:
                    d = v - rows[next_r][p]
                    if d > 0:
                        new_diffs.add(d)
            
            # Check if any new diff conflicts with used_mask
            conflict = False
            for d in new_diffs:
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            
            if not conflict:
                # Also check within-row uniqueness
                all_row_diffs = set()
                for i in range(num_elements):
                    for j in range(i + 1, num_elements):
                        if assigned[next_r][i] and assigned[next_r][j]:
                            d = rows[next_r][j] - rows[next_r][i]
                            if d in all_row_diffs:
                                conflict = True
                                break
                            all_row_diffs.add(d)
                    if conflict:
                        break
            
            if not conflict:
                # Update used_mask
                for d in new_diffs:
                    used_mask |= (1 << d)
                
                # Recurse
                if backtrack(next_r, next_pos + 1):
                    return True
                
                # Undo
                for d in new_diffs:
                    used_mask &= ~(1 << d)
            
            # Undo assignment
            assigned[next_r][next_pos] = False
            rows[next_r][next_pos] = None
        
        return False
    
    if backtrack(0, 1):
        # Verify
        result_rows = [rows[r] for r in range(n_rows)]
        if _is_valid(result_rows):
            return result_rows
        return None
    return None


def _column_wise_construct(n_rows, num_elements, max_scope, rng, deadline, time_check):
    """
    Column-wise construction: assign values column by column (position 1, then 2, etc.)
    across all rows. This breaks the row-by-row dependency.
    """
    if time_check() > deadline:
        return None
    
    rows = [[0] + [None] * (num_elements - 1) for _ in range(n_rows)]
    used_mask = 0
    
    def backtrack(col):
        """Assign column col to all rows."""
        if time_check() > deadline:
            return False
        
        if col >= num_elements:
            return True
        
        # For this column, assign values to all rows
        # Use MRV: start with the most constrained row
        row_order = list(range(n_rows))
        
        # Sort by domain size (smallest first)
        def domain_size(r):
            if col > 0:
                if rows[r][col - 1] is None:
                    return float('inf')
                low = rows[r][col - 1] + 1
            else:
                low = 1
            high = max_scope
            if low > high:
                return 0
            # Quick estimate: count valid values
            count = 0
            for v in range(low, high + 1):
                conflict = False
                for p in range(col):
                    d = v - rows[r][p]
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    count += 1
            return count
        
        row_order.sort(key=lambda r: domain_size(r))
        
        for r in row_order:
            if time_check() > deadline:
                return False
            
            # Get domain for (r, col)
            if col > 0:
                if rows[r][col - 1] is None:
                    return False
                low = rows[r][col - 1] + 1
            else:
                low = 1
            high = max_scope
            
            if low > high:
                return False
            
            # Collect valid values
            valid_vals = []
            for v in range(low, high + 1):
                conflict = False
                # Check against all assigned positions in this row
                for p in range(col):
                    d = v - rows[r][p]
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                # Check within-row diff uniqueness
                if not conflict:
                    new_diffs = set()
                    for p in range(col):
                        d = v - rows[r][p]
                        if d in new_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                
                if not conflict:
                    valid_vals.append(v)
            
            if not valid_vals:
                return False
            
            # Try values in randomized order
            rng.shuffle(valid_vals)
            
            for v in valid_vals:
                if time_check() > deadline:
                    return False
                
                # Assign
                rows[r][col] = v
                
                # Update used_mask
                new_diffs = set()
                for p in range(col):
                    d = v - rows[r][p]
                    new_diffs.add(d)
                
                # Check for conflicts
                conflict = False
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    # Update mask and recurse
                    for d in new_diffs:
                        used_mask |= (1 << d)
                    
                    if backtrack(col):
                        # Continue with next row in this column
                        # But we need to handle the full column assignment
                        pass
                    else:
                        # Undo
                        for d in new_diffs:
                            used_mask &= ~(1 << d)
                    rows[r][col] = None
            
            # If we got here, all values failed for this row
            return False
        
        # All rows in this column assigned, move to next column
        return backtrack(col + 1)
    
    if backtrack(1):
        result_rows = [rows[r] for r in range(n_rows)]
        if _is_valid(result_rows):
            return result_rows
        return None
    return None


def _simulated_annealing(rows, num_elements, rng, time_limit, deadline, time_check):
    """
    Simulated annealing to reduce scope.
    Moves: swap two elements in different rows, or move an element to a new position.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Compute all differences
    def compute_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    current_diffs = compute_all_diffs(current_rows)
    T = 50.0
    cooling = 0.999
    
    while time.time() < min(time_limit, deadline):
        if time_check() > deadline:
            break
        
        # Random move: pick a random element and try to move it
        r = rng.randint(0, len(current_rows) - 1)
        pos = rng.randint(1, num_elements - 1)
        row = current_rows[r]
        old_val = row[pos]
        
        # Try to move to a nearby value
        delta = rng.randint(-10, 10)
        new_val = old_val + delta
        
        # Ensure increasing order
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < num_elements - 1 and new_val >= row[pos + 1]:
            continue
        if new_val < 1:
            continue
        
        # Compute new diffs for this row
        row[pos] = new_val
        new_row_diffs = set()
        valid = True
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                d = row[j] - row[i]
                if d in new_row_diffs:
                    valid = False
                    break
                new_row_diffs.add(d)
            if not valid:
                break
        
        if not valid:
            row[pos] = old_val
            continue
        
        # Check global uniqueness
        # Remove old diffs, add new diffs
        old_row_diffs = set()
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                if i != pos and j != pos:
                    old_row_diffs.add(row[j] - row[i])
        
        # This is complex; let's just recompute
        row[pos] = old_val
        all_diffs = compute_all_diffs(current_rows)
        
        row[pos] = new_val
        new_all_diffs = compute_all_diffs(current_rows)
        
        if len(new_all_diffs) != len(set(new_all_diffs)):
            row[pos] = old_val
            continue
        
        # Compute energy change (scope increase)
        new_scope = _scope(current_rows)
        delta_E = new_scope - current_scope
        
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try CSP-based construction with various scope bounds
    scope_attempts = []
    base = max(counting_bound, 10)
    for offset in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]:
        scope_attempts.append(base + offset)
    scope_attempts.sort()
    
    seen = set()
    unique_attempts = []
    for s in scope_attempts:
        if s not in seen:
            seen.add(s)
            unique_attempts.append(s)
    scope_attempts = unique_attempts
    
    for attempt_idx, target_scope in enumerate(scope_attempts):
        if time.time() > DEADLINE - 50:
            break
        
        # Try multiple seeds
        for seed in range(3):
            if time.time() > DEADLINE - 50:
                break
            
            rng_local = random.Random(seed * 1000 + attempt_idx * 7 + 13)
            
            # Try CSP construction
            rows = _csp_construct(
                n_rows, num_elements, target_scope, rng_local,
                DEADLINE - 50, time.time
            )
            
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                
                if s <= counting_bound + 2:
                    break
        
        if best_rows is not None and best_scope <= counting_bound + 5:
            break
    
    # Phase 2: Try column-wise construction
    if best_rows is None or best_scope > counting_bound + 10:
        for target_scope in scope_attempts[:8]:
            if time.time() > DEADLINE - 40:
                break
            
            for seed in range(2):
                if time.time() > DEADLINE - 40:
                    break
                
                rng_local = random.Random(seed * 500 + target_scope)
                
                rows = _column_wise_construct(
                    n_rows, num_elements, target_scope, rng_local,
                    DEADLINE - 40, time.time
                )
                
                if rows is not None and _is_valid(rows):
                    s = _scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = rows
    
    # Phase 3: Greedy fallback if no solution found
    if best_rows is None:
        for seed in range(5):
            if time.time() > DEADLINE - 30:
                break
            rng_local = random.Random(seed * 31 + 17)
            
            rows = []
            used_mask = 0
            success = True
            
            for row_idx in range(n_rows):
                row = [0]
                for pos in range(1, num_elements):
                    found = False
                    candidates = list(range(row[-1] + 1, row[-1] + 60))
                    rng_local.shuffle(candidates)
                    
                    for val in candidates:
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
                            found = True
                            break
                    
                    if not found:
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
    
    # Final fallback
    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, num_elements)
        best_scope = _scope(best_rows)
    
    # Phase 4: Simulated annealing refinement
    if time.time() < DEADLINE - 20:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15,
            deadline=DEADLINE,
            time_check=time.time
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _greedy_fallback(n_rows, num_elements)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
