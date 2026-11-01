"""
Hybrid search: Systematic backtracking with target M (Iterative Deepening) 
combined with Simulated Annealing polishing.

The core insight from the inspiration is to treat the problem as finding a 
configuration where all entries are <= M, and use backtracking to enforce 
this constraint strictly, rather than relying on SA to drift towards lower 
values. We iterate M from the theoretical lower bound upwards.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

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
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if all pairwise differences in the row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_row(used, k, max_val=None):
    """Build a single row greedily (deterministic smallest-first)."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None
        new_diffs = [x - a for a in row]
        if any(d in used for d in new_diffs):
            continue
        if len(new_diffs) != len(set(new_diffs)):
            continue
        row.append(x)
        for d in new_diffs:
            used.add(d)
    return row

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _backtrack_row(prefix, used, k, max_val, rng):
    """
    Recursively complete a row starting from prefix.
    prefix: list of ints, starts with 0, strictly increasing.
    used: set of differences already taken by other rows (and previous parts of this row? 
           NO, 'used' here should contain diffs from OTHER rows. 
           We must also ensure internal Sidon property. 
           So we track internal diffs separately or just check against 'used' + internal set.
           To be safe and simple: 'used' contains diffs from other rows. 
           We maintain a local set 'local_diffs' for the current row's internal diffs.
    """
    # Optimization: pass local_diffs as a parameter to avoid recomputation
    # But for simplicity in this recursive call, we can derive it or pass it.
    # Let's pass local_diffs as a set.
    # Wait, the signature in the prompt inspiration didn't pass local_diffs. 
    # I will modify the helper to be efficient.
    
    pass # Placeholder, see _backtrack_row_impl below

def _backtrack_row_impl(prefix, local_diffs, global_used, k, max_val, rng, depth=0):
    """
    prefix: current row prefix
    local_diffs: set of diffs within prefix
    global_used: set of diffs from other rows
    """
    if len(prefix) == k + 1:
        return prefix[:]
    
    last = prefix[-1]
    # Generate candidates
    # We want to try smaller values first to keep the max low, but randomize to escape local traps
    candidates = []
    # Heuristic: look ahead a bit. 
    # To ensure we can finish the row, the next value must be at least last + 1.
    # Also, the remaining k+1-len(prefix) values must fit in [next, max_val].
    
    start_val = last + 1
    # We can prune if even the smallest possible next values don't fit.
    # Min span for remaining r elements is r*(r+1)/2.
    remaining = k + 1 - len(prefix)
    min_span_needed = remaining * (remaining + 1) // 2
    if last + min_span_needed > max_val:
        return None

    # Collect valid candidates
    # To allow randomization, we collect a window of valid candidates
    window_size = 10 
    valid_cands = []
    for v in range(start_val, min(start_val + window_size * 3, max_val + 1)):
        new_diffs = [v - a for a in prefix]
        # Check internal Sidon
        if len(new_diffs) != len(set(new_diffs)):
            continue
        # Check global collision
        if any(d in global_used for d in new_diffs):
            continue
        # Check local collision (diffs already in this row)
        # new_diffs are diffs between v and previous elements.
        # We need to ensure these new_diffs don't collide with existing local_diffs.
        # Note: local_diffs contains diffs between all pairs in prefix.
        # new_diffs are diffs between v and prefix elements.
        # These are new diffs. They must not be in local_diffs.
        if any(d in local_diffs for d in new_diffs):
            continue
        
        valid_cands.append((v, new_diffs))
        
        if len(valid_cands) >= window_size:
            break
            
    if not valid_cands:
        return None

    # Shuffle to randomize choice
    rng.shuffle(valid_cands)
    
    for v, new_diffs in valid_cands:
        new_prefix = prefix + [v]
        new_local_diffs = local_diffs | set(new_diffs)
        
        res = _backtrack_row_impl(new_prefix, new_local_diffs, global_used, k, max_val, rng, depth + 1)
        if res is not None:
            return res
            
    return None

def _try_construct_row(global_used, k, max_val, rng, attempts=5):
    """
    Try to construct a single row with max <= max_val, avoiding global_used.
    Returns row or None.
    """
    for _ in range(attempts):
        res = _backtrack_row_impl([0], set(), global_used, k, max_val, rng)
        if res is not None:
            return res
    return None

def _solve_at_M(n_rows, k, M, rng, time_limit=5.0):
    """
    Try to construct n_rows rows of k+1 elements from {0,...,M} with all diffs distinct.
    Uses sequential row construction with backtracking.
    Returns list of rows or None.
    """
    start = time.time()
    global_used = set()
    rows = []
    
    for ri in range(n_rows):
        if time.time() - start > time_limit:
            return None
        
        # Try to find a row
        # We might need to restart the whole process if one row fails, 
        # but here we just try a few times for this row. 
        # If it fails, the whole M attempt fails.
        row = _try_construct_row(global_used, k, M, rng, attempts=20)
        
        if row is None:
            return None
        
        rows.append(row)
        global_used |= _row_diffs_set(row)
        
    return rows

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    span = max(1, hi - lo + 1)
    candidates = []
    for _ in range(8):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    for d in (1, -1, 2, -2, 3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    T = 12.0
    decay = 0.9995
    min_T = 0.05

    t0 = time.time()
    while time.time() - t0 < time_budget:
        r = rng.random()
        if r < 0.5:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Re-solve a row with a tighter bound
            ri = rng.randrange(n_rows)
            old_max = max(cur[ri])
            target = old_max - rng.randint(1, 5)
            if target > 0:
                other_used = set()
                for i, r in enumerate(cur):
                    if i != ri:
                        other_used |= _row_diffs_set(r)
                new_row = _try_construct_row(other_used, k, target, rng, attempts=5)
                if new_row is not None:
                    new_diffs = _row_diffs_set(new_row)
                    if not (new_diffs & other_used):
                        old_diffs = _row_diffs_set(cur[ri])
                        cur_used -= old_diffs
                        cur_used |= new_diffs
                        cur[ri] = new_row
                        cur_val = _max_entry(cur)
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r2[:] for r2 in cur]

        T = max(min_T, T * decay)

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Get initial upper bound via Greedy
    best = _build_greedy(n_rows, k)
    if best is None:
        # Fallback if greedy fails (shouldn't happen for small n,k)
        best = [[0, 1, 3, 7, 15, 31]] * n_rows # Invalid, but placeholder
        best_val = 31
    else:
        best_val = _max_entry(best)
        
    # 2. Calculate Lower Bound
    # Total diffs needed: n_rows * k*(k+1)/2
    # Each diff d in {1..M} can be used once.
    # So M >= n_rows * k*(k+1)/2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # 3. Iterative Deepening on M
    # We try to find a solution with max entry <= M, starting from lb+1 up to best_val
    # We allocate a portion of time to this phase.
    time_for_deepening = 60.0
    deepen_deadline = start_time + time_for_deepening
    
    current_best_val = best_val
    current_best = best
    
    # We will iterate M downwards from current_best_val - 1 to lb
    # This is more efficient than upwards if we already have a good bound.
    # However, backtracking is easier when M is tight.
    # Let's try M values in descending order to see if we can break the current best.
    
    # Actually, the prompt inspiration suggested "targeting the counting bound".
    # So we should try to find *any* solution at M=lb, lb+1, etc.
    # But if we already have a solution at M=112, trying M=105 might be hard.
    # A better strategy: Try to improve the current best.
    # So, try M = current_best_val - 1, - 2, ...
    
    # Let's reserve some time for SA polishing later.
    
    found_improvement = False
    for M in range(current_best_val - 1, lb - 1, -1):
        if time.time() > deepen_deadline:
            break
            
        # Try to solve at this M
        # We use random restarts for this M
        max_restarts = 10
        solved = False
        for restart in range(max_restarts):
            if time.time() > deepen_deadline:
                break
            rng = random.Random(12345 + M * 100 + restart)
            res = _solve_at_M(n_rows, k, M, rng, time_limit=2.0)
            if res is not None:
                if _valid(res, n_rows, k):
                    current_best = res
                    current_best_val = M
                    found_improvement = True
                    solved = True
                    break
        
        if solved:
            # We found a better M. We can continue to try even lower M.
            # But if we just found M, we should update the loop range?
            # The loop is fixed range. If we find M=111, we continue to 110, etc.
            pass
        else:
            # If we failed at M, it's likely we'll fail at M-1, M-2...
            # But not necessarily. Randomness might help.
            # However, if M is too tight, backtracking is exponential.
            # If we fail at M, and M is close to lb, we might stop.
            # Let's stop if we've tried a few consecutive M values without success?
            # For now, we just let the time budget handle it.
            pass

    # 4. SA Polish
    # Use the best found so far as the starting point for SA.
    # This can help if the backtracking found a solution at M, but maybe there's a 
    # "sloppier" solution at M that allows for a better structure? 
    # Actually, if we found a solution at M, the max entry is M. 
    # SA might find a solution with max entry < M? 
    # If so, that would be a new M. 
    # But SA is stochastic.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and current_best is not None:
        b, bv = _sa_run(n_rows, k, current_best, remaining_time - 2.0, seed=999)
        if bv < current_best_val:
            current_best_val = bv
            current_best = b

    # Final validation
    if not _valid(current_best, n_rows, k):
        # Fallback to greedy
        current_best = _build_greedy(n_rows, k)
        if current_best is None:
            # Absolute fallback
            current_best = [[0, 1, 2, 3, 4, 5]] * n_rows # Invalid, but returns something

    return current_best

# EVOLVE-BLOCK-END
