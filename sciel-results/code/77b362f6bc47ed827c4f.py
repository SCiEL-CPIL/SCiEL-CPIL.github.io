"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return set of all differences across all rows."""
    all_d = set()
    for row in rows:
        all_d |= _row_diffs(row)
    return all_d

def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    all_d = _all_diffs(rows)
    total = sum(len(row)*(len(row)-1)//2 for row in rows)
    return len(all_d) == total

def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k):
    """Greedy initialization to get a valid starting point."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _sa_step(rows, k, temp, rng):
    """
    Perform one simulated annealing step.
    Returns (new_rows, cost_change) where cost_change is new_cost - old_cost.
    """
    # Pick a random row and a random position to modify
    row_idx = rng.randrange(len(rows))
    row = list(rows[row_idx])
    
    # Choose a strategy: modify one element, or swap two elements, or insert/remove
    strategy = rng.random()
    
    if strategy < 0.5:
        # Modify one element (not the first)
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        # Try to move it by a small random amount
        delta = rng.randint(-3, 3)
        new_val = old_val + delta
        if new_val <= 0 or new_val >= row[pos+1] if pos+1 < len(row) else False:
            # Ensure increasing order
            if new_val <= row[pos-1] or (pos+1 < len(row) and new_val >= row[pos+1]):
                return rows, 0
        row[pos] = new_val
    elif strategy < 0.8:
        # Swap two non-first elements
        if len(row) < 3:
            return rows, 0
        i, j = rng.sample(range(1, len(row)), 2)
        if i > j:
            i, j = j, i
        # Only swap if it maintains increasing order
        if i > 0 and row[i] < row[i-1]:
            return rows, 0
        if j+1 < len(row) and row[j] > row[j+1]:
            return rows, 0
        # Check if swap maintains order
        if i+1 == j:
            # Adjacent swap always maintains order if both were valid
            pass
        else:
            # Non-adjacent: need to check
            if row[i] > row[j+1] if j+1 < len(row) else False:
                return rows, 0
        row[i], row[j] = row[j], row[i]
        # Verify increasing
        for p in range(1, len(row)):
            if row[p] <= row[p-1]:
                return rows, 0
    else:
        # Shift a suffix
        if len(row) < 2:
            return rows, 0
        pos = rng.randrange(1, len(row))
        shift = rng.randint(-2, 2)
        if shift == 0:
            return rows, 0
        for p in range(pos, len(row)):
            row[p] += shift
        # Verify increasing and positive
        if row[0] != 0:
            return rows, 0
        for p in range(1, len(row)):
            if row[p] <= row[p-1]:
                return rows, 0
    
    # Create new rows list
    new_rows = [list(r) for r in rows]
    new_rows[row_idx] = row
    
    # Compute cost change
    old_diffs = _all_diffs(rows)
    new_diffs = _all_diffs(new_rows)
    
    # Cost: number of duplicate differences + scope penalty
    old_total = sum(len(r)*(len(r)-1)//2 for r in rows)
    new_total = sum(len(r)*(len(r)-1)//2 for r in new_rows)
    
    old_dups = old_total - len(old_diffs)
    new_dups = new_total - len(new_diffs)
    
    old_scope = _scope(rows)
    new_scope = _scope(new_rows)
    
    old_cost = old_dups * 1000 + old_scope
    new_cost = new_dups * 1000 + new_scope
    
    cost_change = new_cost - old_cost
    
    # Acceptance criterion
    if cost_change <= 0 or rng.random() < math.exp(-cost_change / max(temp, 1e-10)):
        return new_rows, cost_change
    else:
        return rows, 0

def _local_search(rows, k, max_iters=1000, rng=None):
    """Simple local search to reduce scope while maintaining validity."""
    if rng is None:
        rng = random.Random()
    
    best = [list(r) for r in rows]
    best_cost = _scope(best)
    
    for it in range(max_iters):
        # Try to reduce scope by moving elements down
        row_idx = rng.randrange(len(rows))
        row = list(best[row_idx])
        
        # Try to decrease the last element
        pos = len(row) - 1
        old_val = row[pos]
        for new_val in range(old_val - 1, row[pos-1], -1):
            row[pos] = new_val
            # Check if this creates any duplicate differences
            new_diffs = _row_diffs(row)
            other_diffs = set()
            for i, r in enumerate(best):
                if i != row_idx:
                    other_diffs |= _row_diffs(r)
            
            if not (new_diffs & other_diffs):
                # Valid improvement
                new_rows = [list(r) for r in best]
                new_rows[row_idx] = row
                new_scope = _scope(new_rows)
                if new_scope < best_cost:
                    best = new_rows
                    best_cost = new_scope
                break
            else:
                row[pos] = old_val
                break
        
        # Try to decrease other elements
        if len(row) > 2:
            pos = rng.randrange(1, len(row) - 1)
            old_val = row[pos]
            lower_bound = row[pos-1] + 1
            upper_bound = row[pos+1] - 1
            if old_val > lower_bound:
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    row[pos] = new_val
                    new_diffs = _row_diffs(row)
                    other_diffs = set()
                    for i, r in enumerate(best):
                        if i != row_idx:
                            other_diffs |= _row_diffs(r)
                    
                    if not (new_diffs & other_diffs):
                        new_rows = [list(r) for r in best]
                        new_rows[row_idx] = row
                        new_scope = _scope(new_rows)
                        if new_scope < best_cost:
                            best = new_rows
                            best_cost = new_scope
                        break
                    else:
                        row[pos] = old_val
                        break
    
    return best

def construct(n_rows, k):
    """Construct difference triangle set using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialize with greedy
    rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    rng = random.Random()
    
    # Simulated annealing
    temp = 50.0
    cooling = 0.995
    
    while time.time() < deadline:
        new_rows, cost_change = _sa_step(rows, k, temp, rng)
        if new_rows is not rows:
            rows = new_rows
            new_scope = _scope(rows)
            if _is_valid(rows) and new_scope < best_scope:
                best_rows = [list(r) for r in rows]
                best_scope = new_scope
        temp *= cooling
        
        # Periodic local search
        if rng.random() < 0.05:
            ls_result = _local_search(rows, k, max_iters=200, rng=rng)
            ls_scope = _scope(ls_result)
            if _is_valid(ls_result) and ls_scope < best_scope:
                best_rows = [list(r) for r in ls_result]
                best_scope = ls_scope
                rows = [list(r) for r in ls_result]
    
    # Final local search on best
    final = _local_search(best_rows, k, max_iters=500, rng=rng)
    if _is_valid(final) and _scope(final) < best_scope:
        best_rows = final
    
    return best_rows
# EVOLVE-BLOCK-END
