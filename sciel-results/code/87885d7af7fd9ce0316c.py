"""Difference triangle set (n_rows, k): rows of k+1 increasing ints from 0,
all intra-row positive differences distinct globally; minimize the largest entry."""
import time
import random
import heapq

_START = time.time()
_DEADLINE = 150.0 - 5.0  # safety margin before the 150s kill

def _deadline():
    return time.time() >= _DEADLINE

def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _greedy_row(k, used):
    """Build a single row greedily: smallest x that doesn't collide with used diffs."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row

def _build_initial(n_rows, k, rng):
    """Build initial solution using greedy with random row ordering."""
    # Shuffle which "slot" gets built first to break symmetry
    order = list(range(n_rows))
    rng.shuffle(order)
    
    used = set()
    rows = [None] * n_rows
    for idx in order:
        row = _greedy_row(k, used)
        rows[idx] = row
    return rows

def _try_shrink_last(rows, used, deadline):
    """Try to reduce the last mark of each row. Returns improved flag."""
    improved = False
    n = len(rows)
    for i in range(n):
        if time.time() > deadline:
            break
        row = rows[i]
        old_last = row[-1]
        old_new = {old_last - a for a in row[:-1]}
        used -= old_new
        for v in range(old_last - 1, row[-2], -1):
            if time.time() > deadline:
                break
            cand_new = {v - a for a in row[:-1]}
            if not (cand_new & used):
                row[-1] = v
                used |= cand_new
                improved = True
                break
        else:
            used |= old_new
    return improved

def _local_search(rows, n_rows, k, deadline):
    """Full local search: try to move interior elements to smaller values."""
    row_diffs = [_row_diffs(r) for r in rows]
    all_used = set()
    for s in row_diffs:
        all_used |= s
    used = set(all_used)
    
    best = max(max(r) for r in rows)
    
    def try_move(r, i, j):
        nonlocal row_diffs, used, all_used
        row = rows[r]
        old = row[j]
        # Differences that involve position j and positions after i
        cur = {row[t] - row[i] for t in range(i + 1, len(row))}
        lo = row[i] + 1
        for v in range(lo, old):
            if v in used:
                continue
            # New differences from v to positions between i+1 and j-1
            nd = {v - row[t] for t in range(i + 1, j)}
            if not (nd & all_used):
                row[j] = v
                row_diffs[r] = (row_diffs[r] - cur) | nd
                used = (used - cur) | nd
                all_used = (all_used - cur) | nd
                return v
        return None
    
    while not _deadline():
        improved_last = _try_shrink_last(rows, used, deadline)
        
        moved = False
        for r in range(n_rows - 1, -1, -1):
            if _deadline():
                break
            row = rows[r]
            j = len(row) - 1
            while j >= 1 and not _deadline():
                i = j - 1
                while i >= 0 and not _deadline():
                    v = try_move(r, i, j)
                    if v is not None:
                        last = row[-1]
                        if last < best:
                            best = last
                            moved = True
                        break
                    i -= 1
                if v is not None:
                    break
                j -= 1
            if _deadline():
                break
        
        if not moved and not improved_last:
            break
    
    return rows, best

def _simulated_annealing(rows, n_rows, k, deadline, rng):
    """SA pass: random perturbations with acceptance based on a cost function."""
    row_diffs = [_row_diffs(r) for r in rows]
    all_used = set()
    for s in row_diffs:
        all_used |= s
    used = set(all_used)
    
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while not _deadline() and T > T_min:
        # Pick a random row and a random position to perturb
        r = rng.randint(0, n_rows - 1)
        row = rows[r]
        j = rng.randint(1, len(row) - 1)
        
        old_val = row[j]
        # Try a random new value
        if j == len(row) - 1:
            new_val = rng.randint(row[j-1] + 1, old_val + 20)
        else:
            new_val = rng.randint(row[j-1] + 1, row[j+1] - 1)
        
        if new_val == old_val:
            continue
        
        # Compute current differences involving position j
        cur_diffs = {row[t] - row[i] for i in range(len(row)) for t in range(i+1, len(row)) 
                     if i == j or t == j}
        
        # Remove old diffs
        used -= cur_diffs
        all_used -= cur_diffs
        row_diffs[r] -= cur_diffs
        
        row[j] = new_val
        
        # Compute new differences
        new_diffs = {row[t] - row[i] for i in range(len(row)) for t in range(i+1, len(row)) 
                     if i == j or t == j}
        
        # Check validity
        if new_diffs & (all_used - new_diffs):
            # Conflict: revert
            row[j] = old_val
            used |= cur_diffs
            all_used |= cur_diffs
            row_diffs[r] |= cur_diffs
            continue
        
        # Accept or reject based on cost
        new_scope = max(max(r2) for r2 in rows)
        old_scope = max(max(r2) for r2 in best_rows) if new_scope >= best_scope else new_scope
        
        # Cost: scope + small penalty for non-optimality
        cost = new_scope
        if cost <= best_scope:
            # Accept
            used |= new_diffs
            all_used |= new_diffs
            row_diffs[r] |= new_diffs
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r2) for r2 in rows]
        else:
            # Probabilistic acceptance
            if rng.random() < (1.0 - cost / (best_scope + T)):
                used |= new_diffs
                all_used |= new_diffs
                row_diffs[r] |= new_diffs
            else:
                # Revert
                row[j] = old_val
                used |= cur_diffs
                all_used |= cur_diffs
                row_diffs[r] |= cur_diffs
        
        T *= cooling
    
    # Restore best found
    for i in range(n_rows):
        rows[i] = list(best_rows[i])
    
    return rows, best_scope

def _priority_queue_greedy(n_rows, k, rng):
    """
    Priority-queue based construction: build rows one at a time, but use a 
    priority queue to decide which "gap" in the difference space to fill next.
    This treats the global difference set as a resource to be carved.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        # Build row greedily but with a twist: when choosing the next element,
        # prefer values that create differences that are "isolated" (few neighbors in used)
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if new & used:
                continue
            
            # Score: prefer x where the new differences are in sparse regions
            # Count how many used values are within ±5 of each new diff
            score = 0
            for d in new:
                for dd in range(max(1, d-5), min(200, d+6)):
                    if dd in used:
                        score += 1
            
            # With some probability, accept a slightly worse x to diversify
            if rng.random() < 0.3 and x < 150:
                # Try a few ahead
                for trial in range(x, min(x + 10, 200)):
                    trial_new = {trial - a for a in row}
                    if trial_new & used:
                        continue
                    trial_score = 0
                    for d in trial_new:
                        for dd in range(max(1, d-5), min(200, d+6)):
                            if dd in used:
                                trial_score += 1
                    if trial_score < score:
                        x = trial
                        new = trial_new
                        score = trial_score
                        break
            
            row.append(x)
            used |= new
        
        rows.append(row)
    
    return rows

def construct(n_rows, k):
    rng = random.Random(42)
    
    # --- Phase 1: Multiple initial constructions ---
    best_rows = None
    best_scope = float('inf')
    
    # Try different strategies for initial construction
    strategies = [
        lambda: _build_initial(n_rows, k, rng),
        lambda: _priority_queue_greedy(n_rows, k, rng),
    ]
    
    for strategy in strategies:
        if _deadline():
            break
        rows = strategy()
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            best_rows.append(_greedy_row(k, used))
        best_scope = max(max(r) for r in best_rows)
    
    # --- Phase 2: Local search ---
    rows, scope = _local_search([list(r) for r in best_rows], n_rows, k, _DEADLINE)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
    
    if _deadline():
        return best_rows
    
    # --- Phase 3: Simulated annealing ---
    rows, scope = _simulated_annealing([list(r) for r in best_rows], n_rows, k, _DEADLINE, rng)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
    
    if _deadline():
        return best_rows
    
    # --- Phase 4: Another local search pass on the SA result ---
    rows, scope = _local_search([list(r) for r in best_rows], n_rows, k, _DEADLINE)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
    
    return best_rows
