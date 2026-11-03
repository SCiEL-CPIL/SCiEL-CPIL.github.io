<<<<<<< SEARCH
"""
Difference triangle set (7,5) — structural rewrite.

Strategy:
  1. Fast greedy initialisation (same as seed) to get a valid baseline.
  2. Backtracking search with forward-checking to find a smaller scope,
     row-by-row, trying to fit each new row within a target scope.
  3. If backtracking within target fails, fall back to the greedy result.
  4. Time-guarded: returns best valid result before the 150 s wall-clock limit.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

_DEADLINE_FRACTION = 0.92  # safety margin: stop at 92 % of 150 s


def _greedy(n_rows, k, seed=0):
    """Greedy ruler construction; returns list of rows."""
    rng = random.Random(seed)
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


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _validate(rows, k):
    """Check that all positive intra-row differences are distinct across rows."""
    seen = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0:
                    return False
                if d in seen:
                    return False
                seen.add(d)
    return True


def _backtrack_row(row, x, k, used, scope, deadline, depth, max_depth):
    """
    Try to extend `row` (already has len(row) entries, last is x) to length k+1,
    using only values <= scope, without reusing any difference in `used`.
    Returns (row, used) on success or None.
    """
    if time.time() > deadline:
        return None
    if len(row) == k + 1:
        return row, used
    if depth > max_depth:
        return None

    for cand in range(x + 1, scope + 1):
        new_diffs = {cand - a for a in row}
        if new_diffs & used:
            continue
        # forward check: ensure there's enough room to place remaining entries
        remaining = (k + 1) - (len(row) + 1)
        if remaining > 0 and scope - cand < remaining:
            break  # not enough room
        new_used = used | new_diffs
        row.append(cand)
        res = _backtrack_row(row, cand, k, new_used, scope, deadline, depth + 1, max_depth)
        row.pop()
        if res is not None:
            return res
    return None


def _try_scope(n_rows, k, target_scope, used, deadline, max_depth=400):
    """
    Try to complete the remaining rows so that every entry <= target_scope
    and no difference is reused.  `used` is the set of differences already
    taken by previously placed rows.  Returns (rows, used) or None.
    """
    rows = []
    current_used = used

    for _ in range(n_rows):
        row = [0]
        res = _backtrack_row(row, 0, k, current_used, target_scope, deadline, 0, max_depth)
        if res is None:
            return None
        row, current_used = res
        rows.append(row)

    return rows, current_used


def _solve_backtracking(n_rows, k, deadline):
    """
    Search for a valid set with minimal scope using backtracking.
    Returns list of rows or None.
    """
    # Lower bound on scope: counting bound
    total_diffs = n_rows * k * (k + 1) // 2
    # Need at least total_diffs distinct positive integers, so scope >= ceil((sqrt(1+8*total_diffs)-1)/2)
    import math
    lb = math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2)

    # Upper bound from greedy
    greedy_rows = _greedy(n_rows, k, seed=0)
    greedy_scope = _scope(greedy_rows)

    # Try scopes from lower bound upward
    for scope in range(lb, greedy_scope + 1):
        if time.time() > deadline:
            break
        # Try to build all rows within this scope
        res = _try_scope(n_rows, k, scope, set(), deadline, max_depth=300)
        if res is not None:
            rows, _ = res
            if _validate(rows, k):
                return rows

    return None


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 150.0 * _DEADLINE_FRACTION

    # 1. Greedy baseline
    best_rows = _greedy(n_rows, k, seed=0)
    best_scope = _scope(best_rows)

    # 2. Try a few greedy seeds to get a better baseline
    for s in range(1, 20):
        if time.time() > deadline:
            break
        rows = _greedy(n_rows, k, seed=s)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows

    # 3. Backtracking search for a smaller scope
    bt_rows = _solve_backtracking(n_rows, k, deadline)
    if bt_rows is not None:
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope

    # 4. Final validation
    if not _validate(best_rows, k):
        # Fallback: regenerate greedy
        best_rows = _greedy(n_rows, k, seed=0)

    return best_rows
=======
"""
Difference triangle set (7,5) — SA-based compression.

Strategy:
  1. Fast greedy initialisation to get a valid baseline.
  2. Simulated annealing: iteratively move individual entries to reduce
     the maximum entry (scope) while maintaining the distinct-difference
     constraint. Uses incremental diff-counting for O(1) validity checks.
  3. Time-guarded: returns best valid result before the 150 s wall-clock limit.
"""

import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

_DEADLINE_FRACTION = 0.92


def _greedy(n_rows, k, seed=0):
    """Greedy ruler construction; returns list of rows."""
    rng = random.Random(seed)
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


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _validate(rows, k):
    """Check that all positive intra-row differences are distinct across rows."""
    seen = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0:
                    return False
                if d in seen:
                    return False
                seen.add(d)
    return True


def _build_diff_counts(rows, k):
    """
    Build a dict mapping each difference to the list of (row_idx, i, j) tuples
    where that difference occurs. Returns (diff_counts, row_diffs) where
    row_diffs[r] is the list of differences in row r.
    """
    diff_counts = defaultdict(list)
    row_diffs = []
    for r_idx, r in enumerate(rows):
        diffs = []
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = r[j] - r[i]
                diffs.append(d)
                diff_counts[d].append((r_idx, i, j))
        row_diffs.append(diffs)
    return diff_counts, row_diffs


def _remove_row_diffs(diff_counts, row_diffs, r_idx):
    """Remove all differences of row r_idx from diff_counts."""
    for d in row_diffs[r_idx]:
        # Find and remove the entry for this row
        lst = diff_counts[d]
        for pos, (ri, i, j) in enumerate(lst):
            if ri == r_idx:
                lst.pop(pos)
                break
        if not lst:
            del diff_counts[d]


def _add_row_diffs(diff_counts, row_diffs, r_idx, row):
    """Add all differences of row r_idx to diff_counts."""
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            d = row[j] - row[i]
            diff_counts[d].append((r_idx, i, j))
    row_diffs[r_idx] = [row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1)]


def _row_rebuild(row, r_idx, k, diff_counts, row_diffs, deadline):
    """
    Try to compress row r_idx by greedily rebuilding it from scratch
    using only the differences not currently used by other rows.
    Returns the new row if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    # Get the set of differences used by all other rows
    other_diffs = set()
    for d, occurrences in diff_counts.items():
        for (ri, i, j) in occurrences:
            if ri != r_idx:
                other_diffs.add(d)
                break
    
    # Remove this row's diffs temporarily
    old_diffs = row_diffs[r_idx][:]
    _remove_row_diffs(diff_counts, row_diffs, r_idx)
    
    # Try to rebuild
    new_row = [0]
    x = 0
    while len(new_row) < k + 1:
        x += 1
        new_diffs = {x - a for a in new_row}
        if new_diffs & other_diffs:
            continue
        new_row.append(x)
        other_diffs |= new_diffs
    
    # Check if we improved
    if max(new_row) < max(row):
        # Update diff_counts and row_diffs
        _add_row_diffs(diff_counts, row_diffs, r_idx, new_row)
        return new_row
    else:
        # Restore original
        _add_row_diffs(diff_counts, row_diffs, r_idx, row)
        return None


def _sa_optimize(rows, k, deadline, seed=0):
    """
    Simulated annealing to reduce the scope (largest entry).
    Moves individual entries to new positions while maintaining validity.
    """
    rng = random.Random(seed)
    n_rows = len(rows)
    
    # Build initial diff structures
    diff_counts, row_diffs = _build_diff_counts(rows, k)
    
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    # SA parameters
    current_temp = 50.0
    cooling = 0.9995
    min_temp = 0.1
    
    while current_temp > min_temp and time.time() < deadline:
        # Pick a random row and a random entry (not the first)
        r_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # positions 1..k (not 0)
        
        old_val = rows[r_idx][pos]
        
        # Try to move this entry to a new position
        # Candidate: try values that might reduce the max
        current_max = max(rows[r_idx])
        
        # Generate candidate values: try to move towards smaller values
        # or try random values within a reasonable range
        candidates = []
        
        # Try slightly smaller values
        for delta in range(1, min(10, old_val)):
            candidates.append(old_val - delta)
        
        # Try slightly larger values (to allow rearrangement)
        for delta in range(1, 10):
            candidates.append(old_val + delta)
        
        # Try random values
        for _ in range(5):
            candidates.append(rng.randint(1, current_max + 20))
        
        # Shuffle candidates
        rng.shuffle(candidates)
        
        improved = False
        for new_val in candidates:
            if new_val <= 0:
                continue
            
            # Check if moving this entry would maintain strictly increasing order
            if pos > 0 and new_val <= rows[r_idx][pos - 1]:
                continue
            if pos < k and new_val >= rows[r_idx][pos + 1]:
                continue
            
            # Compute new differences for this row
            old_row = rows[r_idx][:]
            rows[r_idx][pos] = new_val
            
            # Check validity: all differences in this row must be distinct from others
            new_diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = rows[r_idx][j] - rows[r_idx][i]
                    if d <= 0:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                rows[r_idx] = old_row
                continue
            
            # Check if any new difference conflicts with other rows
            conflict = False
            for d in new_diffs:
                if d in diff_counts:
                    for (ri, i, j) in diff_counts[d]:
                        if ri != r_idx:
                            conflict = True
                            break
                if conflict:
                    break
            
            if conflict:
                rows[r_idx] = old_row
                continue
            
            # Valid move! Update structures
            _remove_row_diffs(diff_counts, row_diffs, r_idx)
            _add_row_diffs(diff_counts, row_diffs, r_idx, rows[r_idx])
            improved = True
            break
        
        if improved:
            current_scope = _scope(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        current_temp *= cooling
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 150.0 * _DEADLINE_FRACTION

    # 1. Greedy baseline - try multiple seeds
    best_rows = None
    best_scope = float('inf')
    
    for s in range(20):
        if time.time() > deadline:
            break
        rows = _greedy(n_rows, k, seed=s)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in rows]

    # 2. SA optimization from the best greedy solution
    if time.time() < deadline:
        sa_rows, sa_scope = _sa_optimize(best_rows, k, deadline, seed=0)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # 3. Try SA from a few other greedy solutions
    for s in [1, 5, 10]:
        if time.time() > deadline:
            break
        rows = _greedy(n_rows, k, seed=s)
        sc = _scope(rows)
        if sc <= best_scope:  # Only try if competitive
            sa_rows, sa_scope = _sa_optimize(rows, k, deadline, seed=s)
            if sa_scope < best_scope:
                best_rows = sa_rows
                best_scope = sa_scope

    # 4. Final validation
    if not _validate(best_rows, k):
        best_rows = _greedy(n_rows, k, seed=0)

    return best_rows
>>>>>>> REPLACE
