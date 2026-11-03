"""Difference triangle set (4,7): construct n_rows rows of k+1 increasing ints starting at 0,
minimizing the largest entry (scope) subject to all positive differences within rows being distinct."""
import time
import random
import sys

def _validate(rows):
    """Return True if rows are valid (strictly increasing, start at 0, all diffs distinct)."""
    used = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _greedy_row(used, k):
    """Build one row greedily: smallest next mark whose diffs are unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _greedy_construct(n_rows, k):
    """Greedy construction of n_rows rows."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _sa_optimize(rows, n_rows, k, deadline, rng):
    """Simulated annealing to reduce scope. Moves: swap two elements within a row, or move an element to a new position."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    cur = [list(r) for r in rows]
    cur_scope = best_scope
    cur_time = time.time()
    start_time = time.time()
    
    # Precompute: for a given set of rows, build the used-diff set
    def _build_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def _check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def _scope_of(rows):
        return max(max(r) for r in rows)
    
    cur_used = _build_used(cur)
    
    # Temperature schedule
    T0 = max(1.0, best_scope * 0.1)
    T_min = 0.01
    steps = 0
    max_steps = 500000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Annealing schedule
        progress = min(1.0, steps / max_steps)
        T = T0 * (T_min / T0) ** progress
        
        # Choose a random move
        move_type = rng.random()
        ri = rng.randrange(n_rows)
        
        if move_type < 0.5:
            # Swap two elements within row ri (not index 0)
            if len(cur[ri]) < 3:
                continue
            i, j = rng.sample(range(1, len(cur[ri])), 2)
            old_i, old_j = cur[ri][i], cur[ri][j]
            cur[ri][i], cur[ri][j] = old_j, old_i
            # Check validity quickly: row must still be strictly increasing
            if any(cur[ri][m] >= cur[ri][m+1] for m in range(len(cur[ri])-1)):
                cur[ri][i], cur[ri][j] = old_i, old_j
                continue
            # Check diff uniqueness
            new_used = set()
            valid = True
            for r in cur:
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        d = r[b] - r[a]
                        if d in new_used:
                            valid = False
                            break
                        new_used.add(d)
                if not valid:
                    break
            if not valid:
                cur[ri][i], cur[ri][j] = old_i, old_j
                continue
            new_scope = _scope_of(cur)
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < _exp_neg(delta / T):
                cur_scope = new_scope
                cur_used = new_used
                if cur_scope < best_scope:
                    best = [list(r) for r in cur]
                    best_scope = cur_scope
            else:
                cur[ri][i], cur[ri][j] = old_i, old_j
        else:
            # Move element: pick a position in row ri, try to change its value
            if len(cur[ri]) < 2:
                continue
            pos = rng.randrange(1, len(cur[ri]))
            old_val = cur[ri][pos]
            # Try nearby values
            offset = rng.choice([-2, -1, 1, 2, 3, -3])
            new_val = old_val + offset
            if new_val <= 0:
                continue
            # Check strictly increasing
            if pos > 0 and new_val <= cur[ri][pos-1]:
                continue
            if pos < len(cur[ri])-1 and new_val >= cur[ri][pos+1]:
                continue
            cur[ri][pos] = new_val
            # Check diff uniqueness
            new_used = set()
            valid = True
            for r in cur:
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        d = r[b] - r[a]
                        if d in new_used:
                            valid = False
                            break
                        new_used.add(d)
                if not valid:
                    break
            if not valid:
                cur[ri][pos] = old_val
                continue
            new_scope = _scope_of(cur)
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < _exp_neg(delta / T):
                cur_scope = new_scope
                cur_used = new_used
                if cur_scope < best_scope:
                    best = [list(r) for r in cur]
                    best_scope = cur_scope
            else:
                cur[ri][pos] = old_val
    
    return best, best_scope

def _exp_neg(x):
    """exp(-x) with clamp to avoid underflow."""
    if x > 50:
        return 0.0
    if x < -50:
        return 1.0
    import math
    return math.exp(-x)

def _local_search(rows, n_rows, k, deadline, rng):
    """Greedy local search: try to reduce the maximum element by re-optimizing the last element of each row."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    cur = [list(r) for r in rows]
    start_time = time.time()
    
    def _build_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def _check_all_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0 or d in used:
                        return False
                    used.add(d)
        return True
    
    # Try to reduce scope by lowering the last element of each row
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            # Try to lower cur[ri][-1]
            last = cur[ri][-1]
            for new_last in range(last - 1, cur[ri][-2], -1):
                cur[ri][-1] = new_last
                if _check_all_valid(cur):
                    new_scope = _scope(cur)
                    if new_scope < best_scope:
                        best = [list(r) for r in cur]
                        best_scope = new_scope
                        improved = True
                        break
            cur[ri][-1] = last  # restore if not improved
    return best, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Greedy initial construction
    rows = _greedy_construct(n_rows, k)
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline:
        sa_rows, sa_scope = _sa_optimize(rows, n_rows, k, deadline, rng)
        if sa_scope < best_scope:
            best = sa_rows
            best_scope = sa_scope
    
    # Phase 3: Local search polish
    if time.time() < deadline - 5:
        ls_rows, ls_scope = _local_search(best, n_rows, k, deadline - 5, rng)
        if ls_scope < best_scope:
            best = ls_rows
            best_scope = ls_scope
    
    # Phase 4: Second SA round with different seed if time remains
    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        sa2_rows, sa2_scope = _sa_optimize(best, n_rows, k, deadline - 10, rng2)
        if sa2_scope < best_scope:
            best = sa2_rows
            best_scope = sa2_scope
    
    # Final validation
    if not _validate(best):
        # Fallback to greedy
        best = _greedy_construct(n_rows, k)
    
    return best
