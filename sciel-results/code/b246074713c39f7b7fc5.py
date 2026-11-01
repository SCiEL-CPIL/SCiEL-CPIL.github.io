"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import copy

def _get_scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _get_all_diffs(rows):
    """Return the set of all positive differences within rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction: each row starts at 0 and takes the smallest
    next mark whose differences to earlier marks are all unused."""
    if rng is None:
        rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _try_reduce_row(row, used_other, rng, max_attempts=200):
    """Try to reduce the largest entry of a single row while keeping it valid
    with respect to used_other (diffs from other rows). Returns (new_row, new_diffs) or None."""
    n = len(row)
    if n < 2:
        return None
    # Try to reduce the last element first, then work backwards
    for attempt in range(max_attempts):
        # Pick a random position to try to reduce (bias towards the end)
        pos = rng.choice(range(1, n))
        old_val = row[pos]
        # Try values from old_val-1 down to row[pos-1]+1
        for new_val in range(old_val - 1, row[pos - 1], -1):
            # Check if new_val - row[i] for i < pos and row[j] - new_val for j > pos
            # are all not in used_other and not in the row's own other diffs
            row_diffs = set()
            valid = True
            for i in range(pos):
                d = new_val - row[i]
                if d in used_other or d in row_diffs:
                    valid = False
                    break
                row_diffs.add(d)
            if not valid:
                continue
            for j in range(pos + 1, n):
                d = row[j] - new_val
                if d in used_other or d in row_diffs:
                    valid = False
                    break
                row_diffs.add(d)
            if not valid:
                continue
            # Also need to check that the diffs involving new_val don't conflict
            # with each other (already done above) and with the rest of the row
            # Build the full new row
            new_row = row[:]
            new_row[pos] = new_val
            # Verify the entire row is strictly increasing
            for i in range(n - 1):
                if new_row[i + 1] <= new_row[i]:
                    valid = False
                    break
            if not valid:
                continue
            # Compute all diffs of new_row and check against used_other
            all_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if d in used_other:
                        valid = False
                        break
                    all_diffs.add(d)
                if not valid:
                    break
            if valid:
                return new_row, all_diffs
        # If we got here, try a slightly different approach: kick a random element
        pos = rng.choice(range(1, n))
        old_val = row[pos]
        # Try a small random perturbation
        for _ in range(10):
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
            if new_val <= row[pos - 1] or (pos < n - 1 and new_val >= row[pos + 1]):
                continue
            new_row = row[:]
            new_row[pos] = new_val
            all_diffs = set()
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if d in used_other:
                        valid = False
                        break
                    all_diffs.add(d)
                if not valid:
                    break
            if valid:
                return new_row, all_diffs
    return None

def _kick(rows, rng, intensity=1):
    """Apply a small random perturbation to the rows."""
    rows = copy.deepcopy(rows)
    for ri in range(len(rows)):
        row = rows[ri]
        n = len(row)
        if n < 2:
            continue
        # Pick a random position (not the first) and nudge it
        if rng.random() < 0.7:
            pos = rng.choice(range(1, n))
            delta = rng.randint(-intensity, intensity)
            new_val = row[pos] + delta
            if new_val > row[pos - 1] and (pos == n - 1 or new_val < row[pos + 1]):
                row[pos] = new_val
    return rows

def _evaluate(rows):
    """Return the scope (largest entry) if valid, else None."""
    if not _is_valid(rows):
        return None
    return _get_scope(rows)

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences are distinct across all rows, minimizing the scope."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    best_rows = _greedy_construct(n_rows, k, rng)
    best_scope = _get_scope(best_rows)
    
    # Phase 2: Local search / ILS to improve
    current_rows = copy.deepcopy(best_rows)
    current_scope = best_scope
    
    # Precompute all diffs for current solution
    current_diffs = _get_all_diffs(current_rows)
    
    max_passes = 50
    pass_count = 0
    no_improve_count = 0
    
    while time.time() < deadline and pass_count < max_passes:
        pass_count += 1
        improved = False
        
        # Try to reduce each row
        for ri in range(n_rows):
            if time.time() > deadline:
                break
            
            # Compute diffs from all other rows
            other_diffs = set()
            for rj in range(n_rows):
                if rj == ri:
                    continue
                row_j = current_rows[rj]
                n_j = len(row_j)
                for i in range(n_j):
                    for j in range(i + 1, n_j):
                        other_diffs.add(row_j[j] - row_j[i])
            
            result = _try_reduce_row(current_rows[ri], other_diffs, rng, max_attempts=100)
            if result is not None:
                new_row, new_diffs = result
                new_scope = max(max(r) for r in current_rows if r is not new_row)
                new_scope = max(new_scope, max(new_row))
                if new_scope < current_scope:
                    current_rows[ri] = new_row
                    current_scope = new_scope
                    improved = True
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = copy.deepcopy(current_rows)
                    break  # Restart row loop after improvement
        
        if not improved:
            no_improve_count += 1
            # Apply a kick to escape local minimum
            current_rows = _kick(current_rows, rng, intensity=2)
            # Re-validate; if invalid, revert to best
            if not _is_valid(current_rows):
                current_rows = copy.deepcopy(best_rows)
                current_scope = best_scope
            # Limit kicks
            if no_improve_count > 30:
                break
        else:
            no_improve_count = 0
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    # Fallback: return greedy
    return _greedy_construct(n_rows, k, random.Random(42))
# EVOLVE-BLOCK-END
