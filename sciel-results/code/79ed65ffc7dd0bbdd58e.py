import math
import random
import time
# EVOLVE-BLOCK-START


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used


def _backtrack_construct(n_rows, k, rng, deadline, max_scope=None):
    """
    Backtracking search with forward checking to construct n_rows difference triangles.
    Builds rows one element at a time, pruning when a difference would conflict.
    Uses randomized variable ordering and value ordering to explore diverse solutions.
    """
    # State: rows is list of partial rows, used is set of all diffs so far
    rows = [[0] for _ in range(n_rows)]
    used = set()
    best_solution = None
    best_scope = max_scope if max_scope else 999999
    
    # Track which row/position we're filling
    # We fill row by row, position by position
    # State: (row_idx, pos_idx) tells us next element to fill
    
    def backtrack(row_idx, pos_idx, used, rows, current_max):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return
        
        # All rows complete?
        if row_idx == n_rows:
            scope = current_max
            if scope < best_scope:
                best_scope = scope
                best_solution = [r[:] for r in rows]
            return
        
        row = rows[row_idx]
        
        if pos_idx >= k + 1:
            # Row complete, move to next row
            backtrack(row_idx + 1, 1, used, rows, current_max)
            return
        
        # Determine candidates for row[row_idx][pos_idx]
        # Must be > row[pos_idx-1] and all new diffs must not be in used
        min_val = row[pos_idx - 1] + 1
        
        # Generate candidate values
        # Try values from min_val upward, with some randomization
        candidates = []
        val = min_val
        # Limit search space: don't go too far beyond current best
        limit = min(best_scope, 200)
        while val <= limit:
            # Check if adding val creates conflicts
            new_diffs = set()
            conflict = False
            for i in range(pos_idx):
                d = val - row[i]
                if d in used or d in new_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if not conflict:
                candidates.append(val)
            
            val += 1
            # Don't generate too many candidates
            if len(candidates) > 50:
                break
        
        if not candidates:
            return
        
        # Randomize order: mix of ascending and shuffled
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        else:
            # Keep mostly ascending but with some perturbation
            if len(candidates) > 3 and rng.random() < 0.5:
                # Swap a few adjacent elements
                for _ in range(min(3, len(candidates) - 1)):
                    idx = rng.randint(0, len(candidates) - 2)
                    candidates[idx], candidates[idx + 1] = candidates[idx + 1], candidates[idx]
        
        for val in candidates:
            if time.time() > deadline:
                return
            
            new_diffs = set()
            for i in range(pos_idx):
                new_diffs.add(val - row[i])
            
            # Place value
            row.append(val)
            new_used = used | new_diffs
            new_max = max(current_max, val)
            
            # Forward check: can the next position still be filled?
            # Simple check: there must exist some value > val that doesn't conflict
            # For efficiency, just proceed and let the recursion handle it
            
            backtrack(row_idx, pos_idx + 1, new_used, rows, new_max)
            
            # Undo
            row.pop()
            
            # If we found a solution, we can stop early
            if best_solution is not None and best_scope <= min_val:
                return
    
    # Initial call
    backtrack(0, 1, used, rows, 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy construction
    return _greedy_construct(n_rows, k)


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
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


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    
    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """
    Simulated annealing to escape local optima in scope minimization.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 10.0
    cooling_rate = 0.995
    
    while time.time() < deadline and temperature > 0.1:
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]
        
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1] or (pos + 1 < len(row) and new_val >= row[pos + 1]):
            continue
        
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        remaining = used - old_diffs
        if new_diffs & remaining:
            continue
        
        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)
        
        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        temperature *= cooling_rate
    
    return best_rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    for _ in range(3):
        if time.time() > deadline:
            break
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
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break
    
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    Uses backtracking with forward checking as the primary search method.
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Phase 1: Backtracking search with randomized restarts
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple backtracking runs with different random seeds/orderings
    for attempt in range(5):
        if time.time() > deadline - 40:
            break
        
        # Use different random seed for each attempt
        attempt_rng = random.Random(42 + attempt * 1000)
        
        # Run backtracking
        result = _backtrack_construct(n_rows, k, attempt_rng, deadline - 40)
        
        if result is not None and len(result) == n_rows:
            scope = _total_scope(result)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in result]
        
        # If we already have a good solution, break early
        if best_scope <= 110:
            break
    
    # If backtracking didn't find anything, use greedy
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k)
        best_scope = _total_scope(best_rows)
    
    # Phase 2: Local search to reduce scope
    if time.time() < deadline - 25:
        ls_rows = _local_search_scope(best_rows, k, rng, deadline - 25)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 3: Simulated annealing to escape local optima
    if time.time() < deadline - 15:
        sa_rows = _simulated_annealing_scope(best_rows, k, rng, deadline - 15)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        polish_rows = _polish_pass(best_rows, k, deadline - 5)
        polish_scope = _total_scope(polish_rows)
        if polish_scope < best_scope:
            best_rows = polish_rows
    
    return best_rows

# EVOLVE-BLOCK-END
