import time
import sys
import random

_DEADLINE = 148.0  # safety margin before 150s kill

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _valid(rows, k):
    """Check that all positive differences are distinct across all rows."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used or d in diffs:
                    return False
                diffs.add(d)
        used |= diffs
    return True

def _fast_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with unused diffs."""
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

def _cp_search(n_rows, k, max_scope, time_limit, rng):
    """
    Constraint Programming search: build rows position by position.
    We assign values to positions (col 1, col 2, ..., col k) for all rows.
    Use forward checking with bitsets for speed.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # We'll assign column by column. For each column j (1..k), assign a value to each row.
    # State: rows[i][j] for i in 0..n_rows-1, j in 0..k (col 0 is always 0)
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    
    # Track used differences globally
    used_diffs = set()
    
    # For forward checking, we need to track what values are feasible for each row at each column
    # Given the increasing constraint and diff constraints
    
    # Strategy: For each column, try to assign values to rows in a smart order.
    # We'll use a recursive backtracking that assigns one (row, col) at a time,
    # but we'll process column by column for better constraint propagation.
    
    # Actually, let's do a smarter approach:
    # Process column by column. For column j, we need to assign rows[i][j] for all i.
    # The value must be > rows[i][j-1] and <= max_scope.
    # And all new differences must not conflict with used_diffs or other new diffs in same column.
    
    # Let's use a different framing: 
    # We have n_rows * k variables (rows[i][j] for j>=1).
    # Constraints:
    # 1. rows[i][j] > rows[i][j-1] for all i, j>=1
    # 2. rows[i][j] <= max_scope
    # 3. All differences rows[i][j] - rows[i][l] for l<j are distinct globally
    
    # Backtracking: assign variables in order (i=0,j=1), (i=1,j=1), ..., (i=n-1,j=1), (i=0,j=2), ...
    # This column-by-column order helps with constraint propagation.
    
    # For efficiency, we'll use a set for used_diffs and check conflicts quickly.
    
    # Let's implement iterative deepening on max_scope, starting from a lower bound.
    # Lower bound: counting argument gives 105 for (7,5).
    
    # Actually, let's just do a single backtracking search with the given max_scope.
    # We'll order variables column by column, and within a column, by row index.
    
    # Variable ordering: for col in 1..k: for row in 0..n_rows-1: assign rows[row][col]
    
    # We need to be careful with the time limit.
    
    # Let's use a recursive function with memoization-like pruning.
    
    # For speed, let's precompute some things.
    
    # Actually, let's try a different approach: 
    # Use the fact that for (7,5), the problem is small enough that a well-optimized
    # backtracker can find solutions quickly if we have good heuristics.
    
    # Heuristic: for each row, the next value should be as small as possible to leave room.
    
    # Let's implement a simple but effective backtracker.
    
    # State: 
    #   rows: 2D list, partially filled
    #   used_diffs: set of all differences used so far
    #   col: current column being assigned (1..k)
    #   row: current row within the column (0..n_rows-1)
    
    # We'll use a global time check to avoid running too long.
    
    def _backtrack(col, row):
        nonlocal rows, used_diffs, deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return False
        
        if col > k:
            # All columns assigned
            return True
        
        if row == n_rows:
            # Move to next column
            return _backtrack(col + 1, 0)
        
        # Assign rows[row][col]
        # Lower bound: rows[row][col-1] + 1
        # Upper bound: max_scope
        # Must not create a difference that's in used_diffs
        
        # Try values from low to high (to find compact solutions)
        # But we can be smarter: try a few random values too for diversity
        
        lower = rows[row][col - 1] + 1
        upper = max_scope
        
        # Generate candidate values
        # For efficiency, try values in increasing order, but with some randomization
        candidates = list(range(lower, upper + 1))
        
        # Shuffle with a bias towards smaller values
        # Actually, let's try a mix: first few small values, then random
        if len(candidates) > 20:
            # Take first 10, then 10 random, then the rest
            small = candidates[:10]
            mid = rng.sample(candidates[10:-10], min(10, max(0, len(candidates) - 20)))
            large = candidates[-10:]
            candidates = small + mid + large
        else:
            rng.shuffle(candidates)
        
        for val in candidates:
            # Check if val creates any conflict
            # Differences: val - rows[row][j] for j in 0..col-1
            conflict = False
            new_diffs = []
            for j in range(col):
                d = val - rows[row][j]
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Also check that new_diffs don't conflict with each other (shouldn't since they're from same row)
            # But they could conflict with each other? No, they're all distinct since rows[row] is strictly increasing.
            
            # Assign
            rows[row][col] = val
            for d in new_diffs:
                used_diffs.add(d)
            
            if _backtrack(col, row + 1):
                return True
            
            # Unassign
            for d in new_diffs:
                used_diffs.discard(d)
        
        return False
    
    if _backtrack(1, 0):
        return rows
    
    return None

def _cp_search_with_idd(n_rows, k, time_limit, rng):
    """
    Iterative deepening on max_scope for CP search.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Lower bound from counting: 105 for (7,5)
    # Upper bound from greedy
    greedy_rows = _fast_greedy(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    # Try scopes from lower bound up to greedy_scope
    lower_bound = 105  # counting bound for (7,5)
    
    for scope in range(lower_bound, greedy_scope + 1):
        if time.time() > deadline - 2:
            break
        
        # Give each scope attempt a portion of the time
        time_per_scope = min(5.0, (deadline - time.time()) / max(1, greedy_scope - scope + 1))
        
        rows = _cp_search(n_rows, k, scope, time_per_scope, rng)
        if rows is not None:
            if _valid(rows, k):
                return rows
    
    # If CP didn't find anything, return greedy
    return greedy_rows

def _local_search(rows, n_rows, k, time_limit, rng):
    """
    Simple local search: try to reduce scope by adjusting individual entries.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    current_rows = [r[:] for r in rows]
    current_scope = best_scope
    
    while time.time() < deadline:
        # Pick a random row and a random position to try to decrease
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # positions 1..k
        
        if current_rows[row_idx][pos] <= 1:
            continue
        
        # Try to decrease this value
        old_val = current_rows[row_idx][pos]
        new_val = old_val - rng.randint(1, min(5, old_val - 1))
        
        if new_val <= current_rows[row_idx][pos - 1]:
            continue
        
        # Check if the new value creates any conflicts
        # Remove old diffs, add new diffs, check validity
        row = current_rows[row_idx]
        
        # Compute old diffs for this row involving position pos
        old_diffs = set()
        for j in range(k + 1):
            if j != pos:
                if j < pos:
                    old_diffs.add(row[pos] - row[j])
                else:
                    old_diffs.add(row[j] - row[pos])
        
        # Compute new diffs
        new_diffs = set()
        for j in range(k + 1):
            if j != pos:
                if j < pos:
                    new_diffs.add(new_val - row[j])
                else:
                    new_diffs.add(row[j] - new_val)
        
        # Check if new_diffs conflict with any other diffs in the entire solution
        # Build set of all diffs except those from this row
        all_diffs = set()
        for i, r in enumerate(current_rows):
            if i == row_idx:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    all_diffs.add(r[b] - r[a])
        
        # Also check within this row, excluding position pos
        for a in range(k + 1):
            for b in range(a + 1, k + 1):
                if a == pos or b == pos:
                    continue
                all_diffs.add(r[b] - r[a])
        
        if new_diffs & all_diffs:
            continue
        
        # Also check that new_diffs don't conflict with each other (they shouldn't)
        if len(new_diffs) != len(new_diffs):
            continue
        
        # Apply the change
        row[pos] = new_val
        new_scope = _scope(current_rows)
        
        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        elif new_scope == current_scope:
            # Accept with some probability
            if rng.random() < 0.5:
                pass  # keep the change
            else:
                row[pos] = old_val  # revert
        else:
            # Revert
            row[pos] = old_val
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all differences distinct across all rows.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Quick greedy baseline
    greedy_rows = _fast_greedy(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    if not _valid(greedy_rows, k):
        return greedy_rows
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Time budget
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        return best_rows
    
    # Try CP search with iterative deepening
    cp_time = min(remaining - 10, 80)
    if cp_time > 5:
        cp_rows = _cp_search_with_idd(n_rows, k, cp_time, rng)
        if cp_rows is not None and _valid(cp_rows, k):
            cp_scope = _scope(cp_rows)
            if cp_scope < best_scope:
                best_scope = cp_scope
                best_rows = cp_rows[:]
        
        # If we hit the lower bound, stop
        if best_scope <= 105:
            return best_rows
    
    # Local search to polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    if remaining > 10:
        ls_time = min(remaining - 5, 50)
        ls_rows = _local_search(best_rows, n_rows, k, ls_time, rng)
        if ls_rows is not None and _valid(ls_rows, k):
            ls_scope = _scope(ls_rows)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows[:]
    
    return best_rows
