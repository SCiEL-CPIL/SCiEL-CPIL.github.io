"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _validate(rows, k):
    """Check all positive differences within rows are distinct across rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, deadline=None):
    """Greedy: build rows one at a time, picking smallest valid next mark."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
        if deadline is not None and time.time() > deadline:
            break
    return rows

def _randomized_greedy(n_rows, k, deadline=None, rng=None):
    """
    Randomized greedy: for each row, choose the next mark from a candidate set
    biased toward smaller values but with some randomness to explore.
    """
    if rng is None:
        rng = random.Random()
    
    used = set()
    rows = []
    max_scope = None
    
    for _ in range(n_rows):
        if deadline is not None and time.time() > deadline:
            break
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try candidates in order with some randomization
            best_x = None
            # First try the smallest valid x (greedy)
            for trial in range(200):
                if trial == 0:
                    cand = x + 1
                else:
                    # Random candidate: sometimes jump ahead
                    jump = rng.randint(1, 5 + trial // 10)
                    cand = x + 1 + jump * rng.randint(1, 3)
                
                new_diffs = {cand - a for a in row}
                if not new_diffs & used:
                    best_x = cand
                    break
            
            if best_x is None:
                # Fallback: linear search
                cand = x + 1
                while True:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        best_x = cand
                        break
                    cand += 1
            
            row.append(best_x)
            used |= {best_x - a for a in row[:-1]}
            x = best_x
        rows.append(row)
    
    if len(rows) < n_rows:
        return None
    return rows

def _local_improve(row, k, used_by_others, deadline=None, rng=None):
    """
    Try to reduce the max entry of a single row by trying alternative
    placements. 'used_by_others' is the set of diffs used by other rows.
    Returns improved row or None.
    """
    if rng is None:
        rng = random.Random()
    
    best_row = list(row)
    best_max = max(row)
    
    for _attempt in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to rebuild this row with a different strategy
        new_row = [0]
        x = 0
        valid = True
        while len(new_row) < k + 1:
            # Find smallest x that works given used_by_others and new_row's own diffs
            found = False
            cand = x + 1
            # Limit search
            for _ in range(200):
                new_diffs = {cand - a for a in new_row}
                # Check against others' used diffs
                if not (new_diffs & used_by_others):
                    # Also check within-row uniqueness (should be automatic if increasing)
                    # But we need all diffs within this row to be unique among themselves
                    # Since row is increasing, diffs are all positive and we just need
                    # them not to collide with each other - but that's guaranteed by construction
                    # if we pick distinct values. Actually no - two different pairs could
                    # give the same diff. Let me check.
                    # For an increasing sequence, a_j - a_i for i<j are all positive.
                    # They could collide: e.g., [0, 3, 6] has diffs 3, 6, 3 -> collision!
                    # So we need to check within-row diff uniqueness too.
                    internal = set()
                    ok = True
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)+1):
                            if j < len(new_row):
                                d = new_row[j] - new_row[i]
                            else:
                                d = cand - new_row[i]
                            if d in internal:
                                ok = False
                                break
                            internal.add(d)
                        if not ok:
                            break
                    if ok and not (internal & used_by_others):
                        new_row.append(cand)
                        found = True
                        break
                cand += 1
            
            if not found:
                valid = False
                break
            x = cand
        
        if valid and max(new_row) < best_max:
            best_row = new_row
            best_max = max(new_row)
    
    return best_row

def _try_improve_all_rows(rows, k, deadline=None, rng=None):
    """Iteratively try to improve each row."""
    rows = [list(r) for r in rows]
    improved = True
    while improved:
        improved = False
        for i in range(len(rows)):
            if deadline is not None and time.time() > deadline:
                return rows
            used_by_others = set()
            for j in range(len(rows)):
                if j == i:
                    continue
                r = rows[j]
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        used_by_others.add(r[b] - r[a])
            
            new_row = _local_improve(rows[i], k, used_by_others, deadline, rng)
            if new_row is not None and max(new_row) < max(rows[i]):
                rows[i] = new_row
                improved = True
    
    return rows

def _simulated_annealing(n_rows, k, deadline, rng=None):
    """
    SA approach: start with greedy solution, then try random perturbations
    (change one entry of one row) and accept if valid and better or with prob.
    """
    if rng is None:
        rng = random.Random()
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k, deadline)
    if len(rows) < n_rows:
        return rows
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Build used set for current solution
    def get_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    current_used = get_used_set(rows)
    current_scope = best_scope
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 2.0:
        # Pick a random row and a random position to perturb
        ri = rng.randrange(n_rows)
        row = rows[ri]
        if len(row) <= 1:
            continue
        pi = rng.randrange(1, len(row))  # don't change position 0
        
        # Remove this entry's contribution
        old_val = row[pi]
        removed_diffs = set()
        for i in range(pi):
            removed_diffs.add(old_val - row[i])
        for j in range(pi+1, len(row)):
            removed_diffs.add(row[j] - old_val)
        
        # Try a new value
        # Candidate: try nearby values
        tried = False
        for _ in range(20):
            if tried:
                break
            if rng.random() < 0.5:
                new_val = old_val + rng.randint(-3, 3)
            else:
                new_val = rng.randint(1, current_scope + 5)
            
            if new_val <= 0:
                continue
            # Check increasing
            if pi > 0 and new_val <= row[pi-1]:
                continue
            if pi < len(row)-1 and new_val >= row[pi+1]:
                continue
            
            # Compute new diffs involving position pi
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row[i])
            for j in range(pi+1, len(row)):
                new_diffs.add(row[j] - new_val)
            
            # Check no collision with remaining used diffs
            remaining_used = current_used - removed_diffs
            if new_diffs & remaining_used:
                continue
            
            # Check within-row uniqueness for new diffs
            # The new diffs must not collide with each other (they won't since
            # new_val is distinct from all other row entries and row is increasing)
            # But we need to make sure the new diffs don't duplicate existing
            # within-row diffs that don't involve position pi.
            # Actually, remaining_used already contains all diffs not involving pi,
            # and we checked new_diffs & remaining_used == empty, so we're good.
            
            # Apply
            row[pi] = new_val
            new_scope = max(row)
            new_total_scope = max(max(r) for r in rows)
            
            delta = new_total_scope - current_scope
            if delta <= 0 or rng.random() < (1.0 if delta < 0 else (temp ** -delta) if delta > 0 else 1.0):
                # Accept
                current_used = (remaining_used | new_diffs)
                current_scope = new_total_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                tried = True
            else:
                # Reject: revert
                row[pi] = old_val
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k, deadline)
    if len(rows) < n_rows:
        return rows
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Try randomized greedy with multiple seeds
    rng = random.Random(42)
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        r = _randomized_greedy(n_rows, k, deadline, rng)
        if r is not None and len(r) == n_rows:
            scope = _scope(r)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(x) for x in r]
    
    # Phase 3: Local improvement
    if time.time() < deadline - 20.0:
        improved = _try_improve_all_rows(best_rows, k, deadline, rng)
        if _validate(improved, k):
            scope = _scope(improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = improved
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_rows = _simulated_annealing(n_rows, k, deadline - 3.0, rng)
        if _validate(sa_rows, k):
            scope = _scope(sa_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in sa_rows]
    
    # Final validation
    if not _validate(best_rows, k):
        # Fallback to greedy
        rows = _greedy_construct(n_rows, k)
        if len(rows) == n_rows:
            return rows
        return best_rows
    
    return best_rows

# EVOLVE-BLOCK-END
