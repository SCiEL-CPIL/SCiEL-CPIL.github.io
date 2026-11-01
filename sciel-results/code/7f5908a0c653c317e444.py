import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
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

def _backtrack_row(row_prefix, used, k, max_val, rng):
    """
    Try to complete a row starting from row_prefix (list of increasing ints starting at 0).
    Returns completed row or None.
    """
    if len(row_prefix) == k + 1:
        return row_prefix[:]
    
    last = row_prefix[-1]
    candidates = []
    for v in range(last + 1, max_val + 1):
        new_diffs = {v - a for a in row_prefix}
        if len(new_diffs) != len(row_prefix):
            continue
        if new_diffs & used:
            continue
        candidates.append(v)
    
    if not candidates:
        return None
    
    rng.shuffle(candidates)
    
    for v in candidates:
        new_diffs = {v - a for a in row_prefix}
        new_row = row_prefix + [v]
        new_used = used | new_diffs
        result = _backtrack_row(new_row, new_used, k, max_val, rng)
        if result is not None:
            return result
    return None

def _randomized_row(forbidden, k, max_val, rng, max_candidates=8):
    """
    Build a row greedily but with random candidate selection among the first few
    valid choices. forbidden is the set of diffs used by other rows.
    Returns row or None.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    for _ in range(k):
        candidates = []
        x = last_val + 1
        # Safety: don't search too far
        search_limit = max_val + 30 if max_val is not None else 200
        while len(candidates) < max_candidates:
            if max_val is not None and x > max_val:
                break
            if x > search_limit:
                break
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden) and not (new_diffs & used_in_row) and len(new_diffs) == len(row):
                candidates.append(x)
            x += 1
        if not candidates:
            return None
        # Weighted toward smaller values for tighter rows
        chosen = random_weighted_pick(candidates, rng)
        row.append(chosen)
        new_diffs = {chosen - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen
    return row

def random_weighted_pick(candidates, rng):
    """Pick from candidates with bias toward smaller values."""
    if len(candidates) == 1:
        return candidates[0]
    # Exponential decay weights: earlier candidates more likely
    weights = [math.exp(-0.15 * i) for i in range(len(candidates))]
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return candidates[i]
    return candidates[-1]

def _try_construct_at_M(n_rows, k, M, rng, time_limit=5.0):
    """
    Try to construct n_rows rows of k+1 elements from {0,...,M} with all diffs distinct.
    """
    start = time.time()
    used = set()
    rows = []
    
    for ri in range(n_rows):
        if time.time() - start > time_limit:
            return None
        best_row = None
        attempts = 20
        for attempt in range(attempts):
            result = _backtrack_row([0], used, k, M, rng)
            if result is not None:
                best_row = result
                break
        if best_row is None:
            return None
        rows.append(best_row)
        used |= _row_diffs_set(best_row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    rng = random.Random(42)
    
    # Start from greedy
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    current_M = best_val
    
    # Phase 1: Try to find solution at lower M values via backtracking
    for target_M in range(max(lb, current_M - 50), current_M):
        if time.time() > deadline - 40:
            break
        found = False
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() > deadline - 40:
                break
            result = _try_construct_at_M(n_rows, k, target_M, rng, time_limit=2.5)
            if result is not None:
                if _valid(result, n_rows, k):
                    best = result
                    best_val = _max_entry(best)
                    found = True
                    break
        if found:
            current_M = best_val
    
    # Phase 2: Simulated annealing with randomized row replacement
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    
    T = 5.0
    cooling = 0.985
    min_T = 0.05
    sa_deadline = deadline - 35  # leave 35s for polish
    
    while time.time() < sa_deadline:
        if T < min_T:
            T = min_T
        
        ri = rng.randrange(n_rows)
        
        # Compute forbidden diffs (all rows except ri)
        row_ri_diffs = _row_diffs_set(cur[ri])
        forbidden = cur_used - row_ri_diffs
        
        old_row = cur[ri]
        old_max_row = old_row[-1]
        
        # Try to find a row with max <= current max (or slightly below for improvement)
        # First try tight: max <= old_max_row - 1
        target_max = old_max_row - 1
        new_row = _randomized_row(forbidden, k, target_max, rng, max_candidates=10)
        
        if new_row is None:
            # Try max <= current global max
            new_row = _randomized_row(forbidden, k, cur_val, rng, max_candidates=10)
        
        if new_row is None:
            # Try without limit
            new_row = _randomized_row(forbidden, k, None, rng, max_candidates=8)
        
        if new_row is None:
            T *= cooling
            continue
        
        # Compute new global max
        new_global_max = max(
            max(r[-1] for r in cur if r is not cur[ri]),
            new_row[-1]
        )
        delta = new_global_max - cur_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            # Accept neutral moves with some probability to explore
            if rng.random() < 0.3:
                accept = True
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            cur_used -= row_ri_diffs
            cur_used |= _row_diffs_set(new_row)
            cur[ri] = new_row
            cur_val = new_global_max
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
        else:
            # Revert (cur unchanged)
            pass
        
        T *= cooling
    
    # Phase 3: Final polish - aggressive row re-optimization with exact backtracking
    # Try to reduce max by re-solving individual rows with tight constraints
    polish_deadline = deadline - 5
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    
    # Sort rows by their max entry, try to reduce the largest ones first
    for _ in range(500):
        if time.time() > polish_deadline:
            break
        if cur_val <= lb:
            break
        
        # Find row with largest max entry
        max_idx = max(range(n_rows), key=lambda i: cur[i][-1])
        old_max = cur[max_idx][-1]
        
        # Try to re-solve this row with max <= old_max - 1
        row_diffs = _row_diffs_set(cur[max_idx])
        forbidden = cur_used - row_diffs
        
        new_row = None
        for attempt in range(30):
            if time.time() > polish_deadline:
                break
            result = _backtrack_row([0], forbidden, k, old_max - 1, rng)
            if result is not None:
                new_row = result
                break
        
        if new_row is not None:
            new_global_max = max(
                max(r[-1] for r in cur if r is not cur[max_idx]),
                new_row[-1]
            )
            if new_global_max < cur_val:
                cur_used -= row_diffs
                cur_used |= _row_diffs_set(new_row)
                cur[max_idx] = new_row
                cur_val = new_global_max
                best_val = cur_val
                best = [r[:] for r in cur]
        else:
            # Try randomized approach too
            new_row = _randomized_row(forbidden, k, old_max - 1, rng, max_candidates=12)
            if new_row is not None:
                new_global_max = max(
                    max(r[-1] for r in cur if r is not cur[max_idx]),
                    new_row[-1]
                )
                if new_global_max < cur_val:
                    cur_used -= row_diffs
                    cur_used |= _row_diffs_set(new_row)
                    cur[max_idx] = new_row
                    cur_val = new_global_max
                    best_val = cur_val
                    best = [r[:] for r in cur]
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best

# EVOLVE-BLOCK-END
