"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _backtrack_row(used_diffs, k, M, rng=None):
    """
    Try to build a single row of k+1 increasing integers starting at 0,
    all <= M, with no difference colliding with used_diffs and no internal
    duplicate differences (Sidon).
    Returns the row as a list, or None if impossible.
    """
    if rng is None:
        rng = random
    
    row = [0]
    used_local = set()
    
    def dfs(idx, last_val):
        if idx == k:
            return True
        # Try next values from last_val+1 to M
        start = last_val + 1
        end = M + 1
        # Randomize order for diversity
        candidates = list(range(start, end))
        rng.shuffle(candidates)
        for v in candidates:
            # Check differences with all previous elements
            new_diffs = set()
            conflict = False
            for i in range(idx):
                d = v - row[i]
                if d in used_diffs or d in used_local:
                    conflict = True
                    break
                new_diffs.add(d)
            if conflict:
                continue
            # Check internal uniqueness (Sidon within this row)
            # new_diffs already checked against used_local, but we need
            # to ensure no two pairs in the full row produce same diff
            # Since we check against used_local (all diffs so far), this is fine
            row.append(v)
            used_local.update(new_diffs)
            if dfs(idx + 1, v):
                return True
            row.pop()
            for d in new_diffs:
                used_local.discard(d)
        return False
    
    if dfs(1, 0):
        return row[:]
    return None

def _resolve_row_given_others(others_diffs, k, M, rng=None):
    """
    Given the set of differences used by other rows, try to construct a valid
    row of k+1 elements starting at 0, max <= M, avoiding all those differences.
    """
    if rng is None:
        rng = random
    return _backtrack_row(others_diffs, k, M, rng)

def _try_construct_at_M(n_rows, k, M, rng=None):
    """
    Try to construct n_rows valid rows, each with k+1 elements starting at 0,
    all <= M, with globally distinct differences.
    Returns list of rows or None.
    """
    if rng is None:
        rng = random
    
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = _resolve_row_given_others(used_diffs, k, M, rng)
        if row is None:
            return None
        rows.append(row)
        used_diffs |= _row_diffs_set(row)
    
    return rows

def _ils_search(n_rows, k, time_limit=90.0, seed=42):
    """
    Iterated Local Search:
    - Start from a greedy solution
    - Perturb by re-solving one or more rows with randomization
    - Local search: try to reduce max entry by re-solving rows at smaller M
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(seed)
    
    # Initial greedy solution
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    # Try to improve by re-solving rows
    cur = [r[:] for r in best]
    cur_val = best_val
    
    # Local search: try to reduce max by re-solving the row(s) with max entry
    def local_search(rows, val, rng):
        rows = [r[:] for r in rows]
        val = _max_entry(rows)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find rows with maximum entry
            max_rows = [i for i, r in enumerate(rows) if max(r) == val]
            # Try re-solving one of them at a slightly smaller M
            for ri in max_rows:
                if time.time() >= deadline:
                    break
                # Try M from val-1 down to val-5
                for trial_M in range(val - 1, max(val - 6, 10), -1):
                    others_diffs = set()
                    for j, r in enumerate(rows):
                        if j != ri:
                            others_diffs |= _row_diffs_set(r)
                    new_row = _resolve_row_given_others(others_diffs, k, trial_M, rng)
                    if new_row is not None:
                        rows[ri] = new_row
                        new_val = _max_entry(rows)
                        if new_val < val:
                            val = new_val
                            improved = True
                            break
                    if improved:
                        break
        return rows, val
    
    # ILS loop
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Perturb: re-solve a random row with randomization
        ri = rng.randrange(n_rows)
        others_diffs = set()
        for j, r in enumerate(cur):
            if j != ri:
                others_diffs |= _row_diffs_set(r)
        
        # Try to re-solve at current max or slightly larger
        trial_M = cur_val + rng.randint(0, 5)
        new_row = _resolve_row_given_others(others_diffs, k, trial_M, rng)
        
        if new_row is not None:
            cur[ri] = new_row
            cur_val = _max_entry(cur)
        
        # Local search
        cur, cur_val = local_search(cur, cur_val, rng)
        
        if cur_val < best_val:
            best_val = cur_val
            best = [r[:] for r in cur]
            no_improve = 0
        else:
            no_improve += 1
    
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    random.seed(42)
    
    # ILS with row-resolution representation
    best = _ils_search(n_rows, k, time_limit=100.0, seed=42)
    best_val = _max_entry(best)
    
    # Ensure validity
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
# EVOLVE-BLOCK-END
