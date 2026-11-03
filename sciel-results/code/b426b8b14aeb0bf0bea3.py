"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _row_diff_set(row):
    """Compute the set of all positive differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Compute the union of all positive differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diff_set(row)
    return used

def _check_valid(rows):
    """Check if rows are valid: all positive differences distinct over all rows."""
    used = set()
    for row in rows:
        rd = _row_diff_set(row)
        if rd & used:
            return False
        used |= rd
    return True

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng, skip_p=0.0):
    """Greedy construction with optional random skip probability."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if skip_p > 0 and rng.random() < skip_p:
                x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _local_improve(rows, n_rows, k, rng, time_limit):
    """Iteratively try to reduce each element to the smallest feasible value.
    Uses efficient incremental diff-set updates for speed."""
    work = [list(r) for r in rows]
    L = k + 1
    
    # Build the global used set
    used_set = set()
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    current_max = _scope(work)
    
    def try_set(r_idx, pos, new_val):
        """Attempt to change work[r_idx][pos] to new_val. Returns True on success."""
        row = work[r_idx]
        # Strictly increasing check
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        if new_val <= 0:
            return False
        
        # Collect old diffs involving pos
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(row[pos] - row[i])
            elif i > pos:
                old_diffs.add(row[i] - row[pos])
        
        # Collect new diffs involving pos
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - row[i])
            elif i > pos:
                new_diffs.add(row[i] - new_val)
        
        # All new diffs must not collide with other diffs
        other = used_set - old_diffs
        if new_diffs & other:
            return False
        
        # Apply
        for d in old_diffs:
            used_set.discard(d)
        for d in new_diffs:
            used_set.add(d)
        row[pos] = new_val
        return True

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Process positions from right to left within each row
        for r_idx in range(n_rows):
            if time.time() >= time_limit:
                break
            for pos in range(L - 1, 0, -1):
                if time.time() >= time_limit:
                    break
                old_val = work[r_idx][pos]
                lower = work[r_idx][pos - 1] + 1
                upper = old_val - 1
                if upper < lower:
                    continue
                found = False
                # Try smallest feasible value first
                for new_val in range(lower, upper + 1):
                    if try_set(r_idx, pos, new_val):
                        if new_val < old_val:
                            improved = True
                            current_max = _scope(work)
                        found = True
                        break
                if not found:
                    # A few random attempts to escape local minima
                    for _ in range(8):
                        if time.time() >= time_limit:
                            break
                        new_val = rng.randint(lower, upper)
                        if new_val == old_val:
                            continue
                        if try_set(r_idx, pos, new_val):
                            if new_val < old_val:
                                improved = True
                                current_max = _scope(work)
                            break
    
    return work, current_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget=30.0):
    """Simulated annealing to minimize scope with efficient diff tracking."""
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    start = time.time()
    L = k + 1
    
    # Precompute row diffs
    current_row_diffs = [_row_diff_set(r) for r in current_rows]
    
    T = 5.0
    cooling = 0.997
    min_T = 0.01
    
    while T > min_T and (time.time() - start) < time_budget:
        # Pick a random row to perturb
        r_idx = rng.randint(0, n_rows - 1)
        row = list(current_rows[r_idx])
        
        # Choose perturbation type
        pert_type = rng.random()
        
        if pert_type < 0.5:
            # Move one element
            pos = rng.randint(1, len(row) - 1)
            delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 10])
            new_val = row[pos] + delta
            
            # Ensure increasing order
            if pos > 0 and new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= 0:
                continue
            row[pos] = new_val
        elif pert_type < 0.8:
            # Try to reduce last element
            if len(row) > 1:
                new_last = row[-1] - rng.randint(1, 6)
                if new_last <= row[-2]:
                    continue
                row[-1] = new_last
            else:
                continue
        else:
            # Small random rebuild of last few elements
            n_tail = min(3, len(row) - 1)
            for pos in range(len(row) - n_tail, len(row)):
                if pos == 0:
                    continue
                delta = rng.randint(-4, 4)
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    new_val = row[pos - 1] + 1
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                row[pos] = new_val
        
        # Check validity: compute new diffs for this row only
        new_row_diff = _row_diff_set(row)
        
        # Check against other rows' diffs
        conflict = False
        for i in range(n_rows):
            if i != r_idx:
                if new_row_diff & current_row_diffs[i]:
                    conflict = True
                    break
        
        if conflict:
            continue
        
        # Accept based on annealing
        other_max = max(max(r) for i, r in enumerate(current_rows) if i != r_idx)
        new_scope = max(max(row), other_max)
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0 or rng.random() < np.exp(-delta_scope / max(T, 0.01)):
            current_rows[r_idx] = row
            current_row_diffs[r_idx] = new_row_diff
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        
        T *= cooling
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope."""
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-start greedy constructions with varying skip probabilities
    # Use a range of skip probabilities to explore different solution landscapes
    skip_probs = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.03, 0.05, 0.02, 0.08]
    for i, sp in enumerate(skip_probs):
        if time.time() > deadline - 70:
            break
        rows = _greedy_construct(n_rows, k, rng, skip_p=sp)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback to simple greedy
        rows = _greedy_construct(n_rows, k, rng, skip_p=0.0)
        return rows
    
    # Phase 2: Aggressive local improvement (the key structural component)
    # This is the most important phase - it does systematic compaction
    time_left = deadline - time.time()
    local_budget = min(40.0, time_left * 0.4)
    if local_budget > 3:
        best_rows, best_scope = _local_improve(best_rows, n_rows, k, rng, time.time() + local_budget)
    
    # Phase 3: Simulated annealing with remaining time
    time_left = deadline - time.time()
    sa_budget = max(0, time_left - 10)
    if sa_budget > 5:
        best_rows, best_scope = _simulated_annealing(
            best_rows, n_rows, k, rng, time_budget=sa_budget
        )
    
    # Phase 4: Final local improvement pass if time remains
    time_left = deadline - time.time()
    if time_left > 5:
        best_rows, best_scope = _local_improve(best_rows, n_rows, k, rng, time.time() + time_left - 3)
    
    # Final validation
    if not _check_valid(best_rows):
        # Fallback
        best_rows = _greedy_construct(n_rows, k, rng, skip_p=0.0)
    
    return best_rows

# EVOLVE-BLOCK-END
