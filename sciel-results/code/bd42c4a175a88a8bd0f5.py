"""Difference triangle set (4,7) – backtracking with constraint propagation."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_diffs(row):
    """Return the set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _rows_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _all_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, max_val=None):
    """Build one row of k+1 entries starting at 0 using greedy smallest-available."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            return None
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _greedy_construct(n_rows, k):
    """Greedy construction: build rows one by one."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _backtrack_row(used, k, max_val):
    """
    Backtracking search for a single row of k+1 entries starting at 0,
    with all entries <= max_val, using differences not in 'used'.
    Returns the row or None if no solution exists.
    """
    row = [0]
    
    def backtrack(start_val):
        if len(row) == k + 1:
            return True
        for x in range(start_val, max_val + 1):
            # Check if x - a for all a in row are not in used
            conflict = False
            for a in row:
                d = x - a
                if d in used:
                    conflict = True
                    break
            if conflict:
                continue
            # Also check that the new differences don't conflict with each other
            # (they can't since all a are distinct and x is fixed)
            row.append(x)
            if backtrack(x + 1):
                return True
            row.pop()
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, deadline):
    """
    Construct rows using backtracking with increasing max_val.
    """
    start_time = time.time()
    
    # First, get a greedy solution to establish an upper bound
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    # Try to find a solution with smaller scope using backtracking
    # Binary search on the scope
    # Lower bound: we need at least k+1 distinct values per row, and differences must be distinct
    # Simple lower bound: k+1 (trivial), but actually we need more
    # For k=7, we need 8 values, and C(8,2)=28 differences per row, so 4*28=112 differences total
    # The differences range from 1 to max_val, so max_val >= 112
    
    low = 112  # counting bound
    high = greedy_scope
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Try values from low upward
    for scope in range(low, high + 1):
        if time.time() > deadline:
            break
        
        used = set()
        rows = []
        success = True
        
        for i in range(n_rows):
            row = _backtrack_row(used, k, scope)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= _all_diffs(row)
        
        if success:
            best_rows = rows
            best_scope = scope
            break
    
    return best_rows, best_scope


def _local_search(rows, k, max_iters=50000, deadline=None):
    """
    Local search: repeatedly pick a row and a position, try shifting entries
    to reduce the global scope while maintaining validity.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    n_rows = len(rows)
    
    def try_move(candidate_rows):
        """Return (valid, scope) for candidate_rows."""
        seen = set()
        scope = 0
        for row in candidate_rows:
            for i in range(len(row)):
                if row[i] > scope:
                    scope = row[i]
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in seen:
                        return False, scope
                    seen.add(d)
        return True, scope
    
    rng = random.Random(42)
    iters = 0
    start = time.time()
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iters += 1
        
        # Pick a random row
        ri = rng.randrange(n_rows)
        row = list(best_rows[ri])
        k1 = len(row)
        
        improved = False
        
        # Strategy 1: shift a single entry (not the first) to a smaller value
        pos = rng.randrange(1, k1)
        orig = row[pos]
        for delta in range(1, min(orig - row[pos-1], 50)):
            candidate = row[pos] - delta
            if candidate <= row[pos - 1]:
                break
            row[pos] = candidate
            ok = True
            for pp in range(1, k1):
                if row[pp] <= row[pp - 1]:
                    ok = False
                    break
            if not ok:
                row[pos] = orig
                continue
            candidate_rows = [best_rows[i] if i != ri else row for i in range(n_rows)]
            valid, scope = try_move(candidate_rows)
            if valid and scope < best_scope:
                best_rows[ri] = row
                best_scope = scope
                improved = True
                break
            row[pos] = orig
        
        # Strategy 2: try to reduce the last entry
        if not improved:
            pos = k1 - 1
            orig = row[pos]
            for delta in range(1, min(orig - row[pos-1], 30)):
                candidate = row[pos] - delta
                row[pos] = candidate
                candidate_rows = [best_rows[i] if i != ri else row for i in range(n_rows)]
                valid, scope = try_move(candidate_rows)
                if valid and scope < best_scope:
                    best_rows[ri] = row
                    best_scope = scope
                    improved = True
                    break
                row[pos] = orig
        
        # Strategy 3: rebuild a random row with backtracking constrained to current scope
        if not improved and iters % 50 == 0:
            used = set()
            for i in range(n_rows):
                if i == ri:
                    continue
                used |= _all_diffs(best_rows[i])
            new_row = _backtrack_row(used, k, best_scope - 1)
            if new_row is not None:
                candidate_rows = [best_rows[i] if i != ri else new_row for i in range(n_rows)]
                valid, scope = try_move(candidate_rows)
                if valid and scope < best_scope:
                    best_rows[ri] = new_row
                    best_scope = scope
    
    return best_rows, best_scope


def _multi_start_greedy(n_rows, k, n_starts=10, deadline=None):
    """Try multiple greedy constructions with different row orderings."""
    best_rows = None
    best_scope = float('inf')
    
    for start in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        used = set()
        order = list(range(n_rows))
        random.shuffle(order)
        row_slots = [None] * n_rows
        for slot in order:
            row = _greedy_row(used, k)
            row_slots[slot] = row
        scope = _scope(row_slots)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in row_slots]
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s limit
    
    # Phase 1: Greedy construction for an initial upper bound
    rows = _greedy_construct(n_rows, k)
    best_scope = _scope(rows)
    
    # Phase 2: Multi-start greedy for diversity
    if time.time() < deadline - 60:
        ms_rows, ms_scope = _multi_start_greedy(n_rows, k, n_starts=15, deadline=deadline - 60)
        if ms_rows is not None and ms_scope < best_scope:
            rows = ms_rows
            best_scope = ms_scope
    
    # Phase 3: Backtracking construction with smaller scope
    if time.time() < deadline - 30:
        bt_rows, bt_scope = _construct_with_backtracking(n_rows, k, deadline - 30)
        if bt_scope < best_scope:
            rows = bt_rows
            best_scope = bt_scope
    
    # Phase 4: Local search to further compact
    if time.time() < deadline - 10:
        rows, ls_scope = _local_search(rows, k, max_iters=100000, deadline=deadline - 5)
        best_scope = ls_scope
    
    return rows

# EVOLVE-BLOCK-END
