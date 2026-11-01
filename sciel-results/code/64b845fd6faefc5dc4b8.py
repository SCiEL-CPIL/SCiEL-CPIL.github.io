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
    """Return the maximum entry across all rows."""
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

def _backtrack_row(row_prefix, used, k, max_val, rng, time_deadline=None):
    """
    Try to complete a row starting from row_prefix (list of increasing ints starting at 0).
    Returns completed row or None.
    """
    if time_deadline is not None and time.time() > time_deadline:
        return None
    
    if len(row_prefix) == k + 1:
        return row_prefix[:]
    
    last = row_prefix[-1]
    # Try candidate next values
    candidates = []
    for v in range(last + 1, max_val + 1):
        # Check Sidon within row
        new_diffs = {v - a for a in row_prefix}
        if len(new_diffs) != len(row_prefix):
            continue
        # Check no collision with used
        if new_diffs & used:
            continue
        candidates.append(v)
    
    if not candidates:
        return None
    
    # Shuffle for randomness
    rng.shuffle(candidates)
    
    for v in candidates:
        if time_deadline is not None and time.time() > time_deadline:
            return None
        new_diffs = {v - a for a in row_prefix}
        new_row = row_prefix + [v]
        new_used = used | new_diffs
        result = _backtrack_row(new_row, new_used, k, max_val, rng, time_deadline)
        if result is not None:
            return result
    return None

def _try_construct_at_M(n_rows, k, M, rng, time_limit=3.0):
    """
    Try to construct n_rows rows of k+1 elements from {0,...,M} with all diffs distinct.
    Uses backtracking with randomization.
    Returns list of rows or None.
    """
    start = time.time()
    deadline = start + time_limit
    used = set()
    rows = []
    
    for ri in range(n_rows):
        if time.time() > deadline:
            return None
        
        # Try multiple random orderings for this row
        best_row = None
        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline:
                return None
            row = [0]
            result = _backtrack_row(row, used, k, M, rng, deadline)
            if result is not None:
                best_row = result
                break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        used |= _row_diffs_set(best_row)
    
    return rows

def _resolve_row_given_others(rows, ri, k, M, rng, used_excl_ri, time_deadline=None):
    """
    Try to re-solve row ri given other rows are fixed.
    used_excl_ri is the set of all diffs from rows other than ri.
    Returns new row or None.
    """
    if time_deadline is not None and time.time() > time_deadline:
        return None
    attempts = 40
    for attempt in range(attempts):
        if time_deadline is not None and time.time() > time_deadline:
            return None
        row = [0]
        result = _backtrack_row(row, used_excl_ri, k, M, rng, time_deadline)
        if result is not None:
            return result
    return None

def _priority_queue_construct(n_rows, k, rng, time_deadline):
    """
    Priority queue based construction: iteratively select the row with the highest 'potential'
    (lowest maximum entry) while dynamically pruning conflicts.
    """
    # Start with greedy to get initial rows
    initial = _build_greedy(n_rows, k)
    best = [r[:] for r in initial]
    best_val = _max_entry(best)
    
    # Build used set
    cur_used = set()
    for r in best:
        cur_used |= _row_diffs_set(r)
    
    # Compute potential for each row (max entry)
    potentials = [max(r) for r in best]
    
    # Priority queue: (potential, row_index)
    # We want to improve the row with the highest potential first
    import heapq
    pq = [(potentials[i], i) for i in range(n_rows)]
    heapq.heapify(pq)
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < time_deadline and no_improve < max_no_improve:
        # Pop the row with highest potential
        if not pq:
            break
        
        pot, ri = heapq.heappop(pq)
        
        # If this row's potential has already been improved, skip
        if pot != potentials[ri]:
            heapq.heappush(pq, (potentials[ri], ri))
            continue
        
        # Compute used excluding this row
        row_ri_diffs = _row_diffs_set(best[ri])
        used_excl = cur_used - row_ri_diffs
        
        # Try to re-solve this row with a tighter max constraint
        old_max = max(best[ri])
        target_max = old_max - 1
        
        new_row = None
        attempts = 20
        for attempt in range(attempts):
            if time.time() > time_deadline:
                break
            result = _backtrack_row([0], used_excl, k, target_max, rng, time_deadline)
            if result is not None:
                new_row = result
                break
        
        if new_row is not None:
            new_max_entry = max(max(r) for r in best if r is not best[ri])
            new_max_entry = max(new_max_entry, max(new_row))
            
            if new_max_entry < best_val:
                # Accept improvement
                cur_used -= row_ri_diffs
                cur_used |= _row_diffs_set(new_row)
                best[ri] = new_row
                best_val = new_max_entry
                potentials[ri] = max(new_row)
                no_improve = 0
            else:
                no_improve += 1
        
        # Re-push with updated potential
        heapq.heappush(pq, (potentials[ri], ri))
    
    return best, best_val

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    rng = random.Random(42)
    
    # Start from greedy
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    # Counting bound lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Phase 1: Try to find solution at lower M values using backtracking
    current_M = best_val
    for target_M in range(max(lb, current_M - 40), current_M):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = _try_construct_at_M(n_rows, k, target_M, rng, time_limit=2.0)
            if result is not None:
                if _valid(result, n_rows, k):
                    best = result
                    best_val = _max_entry(best)
                    found = True
                    break
        
        if found:
            current_M = best_val
            continue
    
    # Phase 2: Priority queue based local search
    best, best_val = _priority_queue_construct(n_rows, k, rng, deadline)
    
    # Phase 3: Additional local search with row-level re-solving
    cur = [r[:] for r in best]
    cur_val = best_val
    
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = rng.randrange(n_rows)
        
        row_ri_diffs = _row_diffs_set(cur[ri])
        used_excl = cur_used - row_ri_diffs
        
        old_max = max(cur[ri])
        target_max = old_max - 1
        
        new_row = None
        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            result = _backtrack_row([0], used_excl, k, target_max, rng, deadline)
            if result is not None:
                new_row = result
                break
        
        if new_row is not None:
            new_max_entry = max(max(r) for r in cur if r is not cur[ri])
            new_max_entry = max(new_max_entry, max(new_row))
            
            if new_max_entry < cur_val:
                cur_used -= row_ri_diffs
                cur_used |= _row_diffs_set(new_row)
                cur[ri] = new_row
                cur_val = new_max_entry
                best = [r[:] for r in cur]
                best_val = cur_val
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best

# EVOLVE-BLOCK-END
