"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # For the specific case (7,5): k+1 = 6 entries per row, 15 differences per row
    # Total differences needed: n_rows * k*(k+1)/2
    # For (7,5): 7 * 15 = 105 differences, so scope must be at least 105
    
    # Greedy baseline first
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _get_scope(best_rows)
    
    # Try to improve with local search / simulated annealing
    if time.time() < deadline - 5.0:
        improved = _local_search(n_rows, k, best_rows, best_scope, deadline)
        if improved is not None:
            best_rows, best_scope = improved
    
    return best_rows


def _get_scope(rows):
    return max(max(row) for row in rows)


def _all_differences(rows):
    """Get the set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
    return diffs


def _validate(rows, n_rows, k):
    """Check if rows are valid: each row has k+1 entries starting at 0, strictly increasing,
    and all positive differences are distinct across all rows."""
    if len(rows) != n_rows:
        return False
    seen_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen_diffs:
                    return False
                seen_diffs.add(d)
    return True


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row by adding the smallest possible next value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows


def _local_search(n_rows, k, initial_rows, initial_scope, deadline):
    """
    Local search / simulated annealing to reduce the scope.
    """
    if not _validate(initial_rows, n_rows, k):
        return None
    
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_scope = initial_scope
    
    # Precompute differences for current state
    all_diffs = _all_differences(rows)
    
    # Simulated annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    rng = random.Random(42)
    
    step = 0
    while time.time() < deadline and T > min_T:
        step += 1
        # Choose a random row and a random position to modify
        row_idx = rng.randrange(n_rows)
        # Try shifting one entry (not the first, which must be 0)
        pos = rng.randrange(1, k + 1)
        
        old_val = rows[row_idx][pos]
        
        # Try a small perturbation
        delta = rng.randint(-5, 5)
        if delta == 0:
            continue
        new_val = old_val + delta
        
        # Check constraints: must be > previous and < next (or be last)
        if pos > 0 and new_val <= rows[row_idx][pos - 1]:
            continue
        if pos < k and new_val >= rows[row_idx][pos + 1]:
            continue
        
        # Compute old diffs for this row
        old_row_diffs = set()
        for i in range(len(rows[row_idx])):
            for j in range(i + 1, len(rows[row_idx])):
                old_row_diffs.add(rows[row_idx][j] - rows[row_idx][i])
        
        # Compute new diffs for this row
        new_row = rows[row_idx][:]
        new_row[pos] = new_val
        new_row_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_row_diffs.add(new_row[j] - new_row[i])
        
        # Check for conflicts with other rows' diffs
        other_diffs = all_diffs - old_row_diffs
        if new_row_diffs & other_diffs:
            # Conflict: reject or accept with probability
            # Count number of conflicts
            conflicts = len(new_row_diffs & other_diffs)
            if rng.random() < (1.0 / (conflicts + 1)) * T / 100.0:
                # Accept anyway (rare)
                pass
            else:
                T *= cooling
                continue
        
        # Apply the move
        rows[row_idx][pos] = new_val
        all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
        
        # Check if scope improved
        current_scope = _get_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    # Final validation
    if _validate(best_rows, n_rows, k):
        return best_rows, best_scope
    
    return None


def _try_polish(rows, n_rows, k, deadline):
    """Try to reduce scope by shifting entries down."""
    if not _validate(rows, n_rows, k):
        return None
    
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(best_rows)
    all_diffs = _all_differences(best_rows)
    rng = random.Random(123)
    
    while time.time() < deadline:
        improved = False
        # Try to shift each non-zero entry down by 1
        for row_idx in range(n_rows):
            for pos in range(1, k + 1):
                old_val = best_rows[row_idx][pos]
                new_val = old_val - 1
                if new_val <= best_rows[row_idx][pos - 1]:
                    continue
                if pos < k and new_val >= best_rows[row_idx][pos + 1]:
                    continue
                
                # Compute old and new diffs
                old_row = best_rows[row_idx]
                old_diffs = set()
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        old_diffs.add(old_row[j] - old_row[i])
                
                new_row = old_row[:]
                new_row[pos] = new_val
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                
                # Apply
                best_rows[row_idx][pos] = new_val
                all_diffs = (all_diffs - old_diffs) | new_diffs
                improved = True
                new_scope = _get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
        
        if not improved:
            break
    
    if _validate(best_rows, n_rows, k):
        return best_rows, best_scope
    return None


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Greedy baseline first
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _get_scope(best_rows)
    
    # Try local search to improve
    if time.time() < deadline - 10.0:
        result = _local_search(n_rows, k, best_rows, best_scope, deadline - 5.0)
        if result is not None:
            best_rows, best_scope = result
    
    # Final polish
    if time.time() < deadline - 2.0:
        result = _try_polish(best_rows, n_rows, k, deadline)
        if result is not None:
            best_rows, best_scope = result
    
    # Validate final result
    if not _validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows
# EVOLVE-BLOCK-END
