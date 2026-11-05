"""Difference triangle set (n_rows, k) – high-performance construction.

Strategy:
  1. Fast greedy with bitset conflict tracking for a quick feasible baseline.
  2. Iterative tightening: repeatedly try to rebuild rows with a tighter scope.
  3. Backtracking with constraint propagation to build rows under a scope bound.
  4. Local search (swap/move) to refine the scope.
  5. Deadline guard to return the best valid solution before the 150 s limit.
"""

import time
import random
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_left(deadline):
    return deadline - time.time()

def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)

def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for r in rows:
        n = len(r)
        diffs = set()
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                d = r[j] - ri
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & seen:
            return False
        seen |= diffs
    return True

def _row_diffs(row):
    """Return the set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _all_diffs_except(rows, row_idx):
    """Return the set of all differences except those in row_idx."""
    seen = set()
    for i, r in enumerate(rows):
        if i == row_idx:
            continue
        n = len(r)
        for a in range(n):
            ri = r[a]
            for b in range(a + 1, n):
                seen.add(r[b] - ri)
    return seen

# ---------------------------------------------------------------------------
# Greedy construction
# ---------------------------------------------------------------------------

def _greedy_row(k, used_diffs, scope_limit=None):
    """
    Build a single row of k+1 increasing integers starting at 0.
    Uses greedy with 'pressure' heuristic: pick the smallest valid candidate,
    but if multiple candidates have the same smallest new max-diff, prefer
    the one that introduces the smallest maximum new difference.
    """
    row = [0]
    while len(row) < k + 1:
        best_candidate = None
        best_score = None
        
        # Start searching from the last element + 1
        start = row[-1] + 1
        end = scope_limit if scope_limit is not None else start + 200
        
        for x in range(start, end + 1):
            # Check if x creates any conflict
            new_diffs = []
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Pressure: minimize the maximum new difference
            max_new_diff = max(new_diffs)
            
            if best_score is None or max_new_diff < best_score:
                best_score = max_new_diff
                best_candidate = x
                
                # Early stop: if we found a candidate with very small max diff,
                # it's likely optimal for this step
                if max_new_diff == x - row[-1]:
                    break
        
        if best_candidate is None:
            return None
        
        row.append(best_candidate)
        for a in row[:-1]:
            used_diffs.add(best_candidate - a)
    
    return row

def _backtrack_row(k, used_diffs, scope_limit):
    """
    Backtracking search to build a row of k+1 integers in [0, scope_limit]
    with no difference conflicts.
    """
    row = [0]
    
    def backtrack(pos):
        if pos == k + 1:
            return True
        
        start = row[-1] + 1
        end = min(scope_limit, row[-1] + k + 1)  # heuristic bound
        
        for x in range(start, end + 1):
            # Check conflicts
            new_diffs = []
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Try this candidate
            row.append(x)
            for d in new_diffs:
                used_diffs.add(d)
            
            if backtrack(pos + 1):
                return True
            
            # Undo
            row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return False
    
    if backtrack(1):
        return list(row)
    return None

def _construct_diverse(n_rows, k, scope_limit=None, rng=None):
    """
    Construct n_rows rows with diversity: shuffle row order and use
    different greedy strategies to avoid symmetric traps.
    """
    if rng is None:
        rng = random.Random(42)
    
    used_diffs = set()
    rows = []
    
    # Try multiple random orderings
    for attempt in range(10):
        rows = []
        used_diffs = set()
        success = True
        
        # Randomize row construction order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for _ in row_order:
            # Try greedy first
            row = _greedy_row(k, used_diffs, scope_limit)
            if row is None:
                # Try backtracking
                row = _backtrack_row(k, used_diffs, scope_limit)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success:
            return rows
    
    return None

# ---------------------------------------------------------------------------
# Local search: try to reduce scope by moving individual entries
# ---------------------------------------------------------------------------

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to reduce the scope. Tries to move individual entries
    to smaller values if possible.
    """
    cur_rows = [list(r) for r in rows]
    cur_scope = _scope(cur_rows)
    best_rows = [list(r) for r in cur_rows]
    best_scope = cur_scope
    
    # Precompute global used diffs
    def get_global_used(rs):
        used = set()
        for r in rs:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    used.add(r[j] - ri)
        return used
    
    global_used = get_global_used(cur_rows)
    
    while _time_left(deadline) > 0.1:
        improved = False
        
        # Try to reduce the maximum entry
        max_val = 0
        max_r, max_c = 0, 0
        for r in range(n_rows):
            for c in range(1, len(cur_rows[r])):
                if cur_rows[r][c] > max_val:
                    max_val = cur_rows[r][c]
                    max_r, max_c = r, c
        
        row = cur_rows[max_r]
        ci = max_c
        old_val = row[ci]
        
        min_allowed = row[ci - 1] + 1
        max_allowed = row[ci + 1] - 1 if ci < len(row) - 1 else old_val - 1
        
        # Try values from old_val - 1 down to min_allowed
        for new_val in range(min(max_allowed, old_val - 1), min_allowed - 1, -1):
            # Check internal row conflicts
            new_diffs = set()
            for i in range(len(row)):
                if i == ci:
                    continue
                if i < ci:
                    d = new_val - row[i]
                else:
                    d = row[i] - new_val
                new_diffs.add(d)
            
            # Check for duplicates within the row (excluding those involving ci)
            other_diffs_row = set()
            for i in range(len(row)):
                if i == ci:
                    continue
                for j in range(len(row)):
                    if j == ci or j <= i:
                        continue
                    other_diffs_row.add(row[j] - row[i])
            
            if new_diffs & other_diffs_row:
                continue
            
            # Check global conflicts
            old_diffs = set()
            for i in range(len(row)):
                if i == ci:
                    continue
                if i < ci:
                    d = old_val - row[i]
                else:
                    d = row[i] - old_val
                old_diffs.add(d)
            
            conflict = False
            for d in new_diffs:
                if d in global_used and d not in old_diffs:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Valid move!
            row[ci] = new_val
            global_used -= old_diffs
            global_used |= new_diffs
            
            new_scope = _scope(cur_rows)
            if new_scope < cur_scope:
                cur_scope = new_scope
                improved = True
                break
        
        if not improved:
            # Try random perturbations
            ri = rng.randrange(n_rows)
            ci = rng.randrange(1, len(cur_rows[ri]))
            row = cur_rows[ri]
            old_val = row[ci]
            
            min_allowed = row[ci - 1] + 1
            max_allowed = row[ci + 1] - 1 if ci < len(row) - 1 else cur_scope
            
            if max_allowed > min_allowed:
                new_val = rng.randint(min_allowed, max_allowed)
                if new_val != old_val:
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == ci:
                            continue
                        if i < ci:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        new_diffs.add(d)
                    
                    other_diffs_row = set()
                    for i in range(len(row)):
                        if i == ci:
                            continue
                        for j in range(len(row)):
                            if j == ci or j <= i:
                                continue
                            other_diffs_row.add(row[j] - row[i])
                    
                    if not (new_diffs & other_diffs_row):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == ci:
                                continue
                            if i < ci:
                                d = old_val - row[i]
                            else:
                                d = row[i] - old_val
                            old_diffs.add(d)
                        
                        conflict = False
                        for d in new_diffs:
                            if d in global_used and d not in old_diffs:
                                conflict = True
                                break
                        if not conflict:
                            row[ci] = new_val
                            global_used -= old_diffs
                            global_used |= new_diffs
                            new_scope = _scope(cur_rows)
                            if new_scope < cur_scope:
                                cur_scope = new_scope
                                improved = True
                            elif new_scope == cur_scope and rng.random() < 0.3:
                                improved = True
                            else:
                                # Revert
                                row[ci] = old_val
                                global_used -= new_diffs
                                global_used |= old_diffs
        
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [list(r) for r in cur_rows]
        
        if not improved and rng.random() < 0.1:
            # Occasional restart with a small perturbation
            break
    
    return best_rows, best_scope

# ---------------------------------------------------------------------------
# Iterative tightening
# ---------------------------------------------------------------------------

def _iterative_tightening(rows, n_rows, k, deadline, rng):
    """
    Repeatedly try to rebuild rows with a tighter scope.
    """
    cur_rows = [list(r) for r in rows]
    cur_scope = _scope(cur_rows)
    best_rows = [list(r) for r in cur_rows]
    best_scope = cur_scope
    
    while _time_left(deadline) > 1.0:
        # Try to reduce scope by 1
        new_scope = cur_scope - 1
        if new_scope < k:  # minimum possible
            break
        
        # Try to reconstruct with tighter scope
        success = False
        for attempt in range(5):
            new_rows = _construct_diverse(n_rows, k, new_scope, rng)
            if new_rows is not None and _validate(new_rows):
                cur_rows = new_rows
                cur_scope = new_scope
                success = True
                break
        
        if success:
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(r) for r in cur_rows]
        else:
            # Try local search to improve current solution
            if _time_left(deadline) > 2.0:
                ls_rows, ls_scope = _local_search(cur_rows, n_rows, k, deadline, rng)
                if ls_scope < best_scope:
                    best_scope = ls_scope
                    best_rows = ls_rows
                    cur_rows = ls_rows
                    cur_scope = ls_scope
            break
    
    return best_rows, best_scope

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences across all rows are distinct. Minimize the scope.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145 s to leave 5 s margin before 150 s kill

    rng = random.Random(42)

    # Phase 1: Quick greedy to get a valid baseline
    rows = _construct_diverse(n_rows, k, None, rng)
    if rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                    new.add(d)
                if not conflict:
                    row.append(x)
                    used |= new
            rows.append(row)
    
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Iterative tightening
    if _time_left(deadline) > 5.0:
        it_rows, it_scope = _iterative_tightening(best_rows, n_rows, k, deadline, rng)
        if it_scope < best_scope:
            best_rows = it_rows
            best_scope = it_scope

    # Phase 3: Final local search
    if _time_left(deadline) > 2.0:
        ls_rows, ls_scope = _local_search(best_rows, n_rows, k, deadline, rng)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Final validation
    if not _validate(best_rows):
        # Fallback to simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                    new.add(d)
                if not conflict:
                    row.append(x)
                    used |= new
            best_rows.append(row)

    return best_rows
