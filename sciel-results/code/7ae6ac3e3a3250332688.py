"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
import sys

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs_set(row):
    """Compute set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _row_diffs_list(row):
    """Compute list of all positive differences in a row."""
    lst = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            lst.append(row[j] - row[i])
    return lst


def _greedy_row(used, k, rng=None, strategy="pure"):
    """
    Build one row greedily.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 60
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 80):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 12:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            
            # Score by how many future positions remain open
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:8]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 30):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 5:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _backtrack_row(used, k, max_val, rng=None, deadline_check=None):
    """
    Backtracking search for a row of k+1 elements starting at 0,
    with all elements <= max_val, and no conflicts with 'used'.
    Returns the row or None.
    """
    if deadline_check and deadline_check() < 2.0:
        return None
    
    row = [0]
    row_diffs = set()
    
    def backtrack(pos, current_max):
        if deadline_check and deadline_check() < 2.0:
            return None
        if pos == k + 1:
            return row[:]
        
        # Try candidates from current_max+1 to max_val
        start = current_max + 1
        for cand in range(start, max_val + 1):
            new_diffs = {cand - a for a in row}
            if new_diffs & used or new_diffs & row_diffs:
                continue
            
            # Check internal consistency (should be fine since row is sorted)
            row.append(cand)
            row_diffs |= new_diffs
            
            result = backtrack(pos + 1, cand)
            if result is not None:
                return result
            
            row.pop()
            row_diffs -= new_diffs
        
        return None
    
    return backtrack(1, 0)


def _iterative_deepening_row(used, k, start_max, rng=None, deadline_check=None):
    """
    Iteratively increase max_val until we find a valid row.
    """
    max_val = start_max
    while deadline_check is None or deadline_check() > 2.0:
        row = _backtrack_row(used, k, max_val, rng, deadline_check)
        if row is not None:
            return row
        max_val += 5
        if max_val > 500:
            break
    return None


def _construct_with_backtracking(n_rows, k, rng, deadline_check=None):
    """
    Construct rows using backtracking for each row, with iterative deepening.
    """
    used = set()
    rows = []
    
    for r in range(n_rows):
        if deadline_check and deadline_check() < 10.0:
            break
        
        # Start with a reasonable estimate
        est = int((r + 1) * (k * (k + 1) / 2) ** 0.5) + k + 2
        est = max(est, k + 2)
        
        row = _iterative_deepening_row(used, k, est, rng, deadline_check)
        if row is None:
            # Fallback to greedy
            row = _greedy_row(used, k, rng, "pure")
        
        rows.append(row)
        used |= _row_diffs_set(row)
    
    return rows


def _local_search_reduce_scope(rows, n_rows, k, rng, max_iters=100000, deadline_check=None):
    """
    Local search to reduce the scope (largest entry).
    We try to move individual elements to smaller values while maintaining validity.
    """
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    no_improve = 0
    
    for it in range(max_iters):
        if deadline_check and deadline_check() < 5.0:
            break
        if no_improve > 5000:
            break
        
        # Pick a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random position (not 0)
        pos = rng.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try to reduce
        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val - 1
        
        if upper < lower:
            no_improve += 1
            continue
        
        # Try values from lower to min(upper, old_val - 1)
        found = False
        for new_val in range(lower, min(upper, old_val - 1) + 1):
            if new_val >= old_val:
                break
            row[pos] = new_val
            if _validate(rows):
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    no_improve = 0
                else:
                    no_improve += 1
                found = True
                break
            row[pos] = old_val
        
        if not found:
            no_improve += 1
    
    return rows, best_scope


def _simulated_annealing(rows, n_rows, k, rng, time_budget=30.0, deadline_check=None):
    """
    Simulated annealing to minimize scope.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    
    T = 200.0
    cooling = 0.999
    
    start_time = time.time()
    while time.time() - start_time < time_budget and T > 0.1:
        if deadline_check and deadline_check() < 5.0:
            break
        
        # Pick a random row and position
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 30
        
        if upper <= lower:
            T *= cooling
            continue
        
        # Propose new value
        if rng.random() < 0.5:
            new_val = old_val + rng.choice([-3, -2, -1, 1, 2, 3])
        else:
            new_val = rng.randint(lower, min(upper, old_val + 20))
        
        if new_val < lower or new_val > upper:
            T *= cooling
            continue
        
        row[pos] = new_val
        
        # Check validity
        if _validate(current_rows):
            new_scope = _scope(current_rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or rng.random() < math.exp(-max(delta, 1) / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                row[pos] = old_val
        else:
            row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_scope


def _multi_strategy_construct(n_rows, k, rng):
    """
    Try multiple construction strategies and return the best.
    """
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, None, "pure")
        rows.append(row)
        used |= _row_diffs_set(row)
    scope = _scope(rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = [r[:] for r in rows]
    
    # Strategy 2: Shuffled greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, "shuffled")
        rows.append(row)
        used |= _row_diffs_set(row)
    scope = _scope(rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = [r[:] for r in rows]
    
    # Strategy 3: Lookahead greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, "lookahead")
        rows.append(row)
        used |= _row_diffs_set(row)
    scope = _scope(rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = [r[:] for r in rows]
    
    # Strategy 4: Backtracking with iterative deepening
    if _time_left() > 40.0:
        rows_bt = _construct_with_backtracking(n_rows, k, rng, _time_left)
        if rows_bt and len(rows_bt) == n_rows:
            scope = _scope(rows_bt)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows_bt]
    
    # More randomized restarts
    for attempt in range(20):
        if _time_left() < 30.0:
            break
        strategy = rng.choice(["pure", "shuffled", "lookahead"])
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, strategy)
            rows.append(row)
            used |= _row_diffs_set(row)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    return best_rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(12345)

    # Phase 1: Multi-strategy construction
    best_rows, best_scope = _multi_strategy_construct(n_rows, k, rng)
    
    if best_rows is None or len(best_rows) != n_rows:
        # Fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, None, "pure")
            best_rows.append(row)
            used |= _row_diffs_set(row)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search to reduce scope
    if _time_left() > 30.0:
        rows_ls, scope_ls = _local_search_reduce_scope(
            best_rows, n_rows, k, rng, 
            max_iters=50000, deadline_check=_time_left
        )
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
    
    # Phase 3: Simulated annealing
    if _time_left() > 25.0:
        rows_sa, scope_sa = _simulated_annealing(
            best_rows, n_rows, k, rng,
            time_budget=min(20.0, _time_left() - 15.0),
            deadline_check=_time_left
        )
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    # Phase 4: Another local search pass
    if _time_left() > 15.0:
        rows_ls2, scope_ls2 = _local_search_reduce_scope(
            best_rows, n_rows, k, rng,
            max_iters=30000, deadline_check=_time_left
        )
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2
    
    # Final validation
    if not _validate(best_rows):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, None, "pure")
            rows.append(row)
        return rows

    return best_rows
