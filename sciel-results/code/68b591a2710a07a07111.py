import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k):
    """Greedy initialization: each row takes the smallest next mark with no difference collision."""
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if len(new_diffs) != len(row):
                continue
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _gen_valid_rows(k, max_val, limit=200000):
    """
    Generate all valid rows (k+1 increasing integers starting at 0, all <= max_val)
    with no internal difference collisions.
    Returns list of (row_tuple, diffs_set).
    """
    valid_rows = []
    
    def backtrack(pos, current, diffs):
        if len(valid_rows) >= limit:
            return
        if pos == k + 1:
            valid_rows.append((tuple(current), frozenset(diffs)))
            return
        if pos == 0:
            current = [0]
            backtrack(1, current, set())
            return
        
        prev = current[-1]
        remaining = k + 1 - pos
        min_next = prev + 1
        
        # Upper bound: need remaining-1 more elements, each at least +1
        max_next = max_val - (remaining - 1)
        
        for x in range(min_next, max_next + 1):
            new_diffs = set()
            ok = True
            for a in current:
                d = x - a
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                current.append(x)
                diffs |= new_diffs
                backtrack(pos + 1, current, diffs)
                diffs -= new_diffs
                current.pop()
    
    backtrack(0, [], set())
    return valid_rows

def _row_swap_local_search(rows, n_rows, k, time_budget, valid_rows_cache=None):
    """
    Local search where each move replaces an entire row with another valid row
    that doesn't conflict with other rows. Objective: minimize global max.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_rows = [list(r) for r in rows]
    current_diffs = [_row_diffs(r) for r in current_rows]
    current_val = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_val = current_val
    
    # Precompute: for each row, its max entry
    row_maxes = [max(r) for r in current_rows]
    
    # Build a pool of candidate rows sorted by their max entry
    if valid_rows_cache is None:
        # Generate candidate rows up to current_val + some margin
        max_gen = current_val + 30
        valid_rows_cache = _gen_valid_rows(k, max_gen, limit=50000)
    
    # Index valid rows by their max value for efficient lookup
    rows_by_max = defaultdict(list)
    for row_t, diffs_f in valid_rows_cache:
        rows_by_max[max(row_t)].append((row_t, diffs_f))
    
    no_improve = 0
    max_no_improve = 5000
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row to modify
        ri = random.randrange(n_rows)
        old_row = current_rows[ri]
        old_diffs = current_diffs[ri]
        old_max = row_maxes[ri]
        
        # Find other rows' diffs (everything except row ri)
        other_diffs = set()
        for j in range(n_rows):
            if j != ri:
                other_diffs |= current_diffs[j]
        
        # Find candidate rows that:
        # 1. Don't conflict with other_diffs
        # 2. Have max < old_max (to improve) or max == old_max (for diversity)
        
        target_max = old_max - 1
        found_improvement = False
        
        # Try to find a row with smaller max
        for tm in range(max(0, target_max), -1, -1):
            if tm > old_max:
                continue
            candidates = rows_by_max.get(tm, [])
            random.shuffle(candidates)
            for row_t, diffs_f in candidates[:50]:
                if diffs_f & other_diffs:
                    continue
                # Valid replacement
                new_row = list(row_t)
                new_max = tm
                
                # Check if this improves global max
                new_row_maxes = row_maxes[:]
                new_row_maxes[ri] = new_max
                new_global_max = max(new_row_maxes)
                
                if new_global_max < current_val:
                    # Accept improvement
                    current_rows[ri] = new_row
                    current_diffs[ri] = set(diffs_f)
                    row_maxes[ri] = new_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [list(r) for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try swaps: exchange two rows
            if n_rows >= 2:
                rj = random.randrange(n_rows)
                if rj != ri:
                    old_row_j = current_rows[rj]
                    old_diffs_j = current_diffs[rj]
                    old_max_j = row_maxes[rj]
                    
                    # Other diffs excluding both ri and rj
                    other_diffs2 = set()
                    for j in range(n_rows):
                        if j != ri and j != rj:
                            other_diffs2 |= current_diffs[j]
                    
                    # Find replacement for ri that doesn't conflict with rj's diffs and other_diffs2
                    for tm in range(max(0, old_max - 1), -1, -1):
                        candidates = rows_by_max.get(tm, [])
                        random.shuffle(candidates)
                        for row_t, diffs_f in candidates[:30]:
                            if diffs_f & other_diffs2:
                                continue
                            if diffs_f & old_diffs_j:
                                continue
                            
                            # Also find replacement for rj
                            for tm2 in range(max(0, old_max_j - 1), -1, -1):
                                candidates2 = rows_by_max.get(tm2, [])
                                random.shuffle(candidates2)
                                for row_t2, diffs_f2 in candidates2[:30]:
                                    if diffs_f2 & other_diffs2:
                                        continue
                                    if diffs_f2 & diffs_f:
                                        continue
                                    
                                    new_row_maxes = row_maxes[:]
                                    new_row_maxes[ri] = tm
                                    new_row_maxes[rj] = tm2
                                    new_global_max = max(new_row_maxes)
                                    
                                    if new_global_max < current_val:
                                        current_rows[ri] = list(row_t)
                                        current_diffs[ri] = set(diffs_f)
                                        row_maxes[ri] = tm
                                        current_rows[rj] = list(row_t2)
                                        current_diffs[rj] = set(diffs_f2)
                                        row_maxes[rj] = tm2
                                        current_val = new_global_max
                                        
                                        if current_val < best_val:
                                            best_val = current_val
                                            best_rows = [list(r) for r in current_rows]
                                            no_improve = 0
                                        else:
                                            no_improve += 1
                                        found_improvement = True
                                        break
                                if found_improvement:
                                    break
                        if found_improvement:
                            break
                    
                    if found_improvement:
                        continue
            
            no_improve += 1
        
        if no_improve >= max_no_improve:
            restarts += 1
            if restarts >= max_restarts:
                break
            # Restart from best
            current_rows = [list(r) for r in best_rows]
            current_diffs = [_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            current_val = best_val
            no_improve = 0
    
    return best_rows, best_val

def _hybrid_local_search(rows, n_rows, k, time_budget):
    """
    Hybrid: combine element-level moves with row-level swaps.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_rows = [list(r) for r in rows]
    current_diffs = [_row_diffs(r) for r in current_rows]
    current_val = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_val = current_val
    row_maxes = [max(r) for r in current_rows]
    
    # Generate valid row pool
    max_gen = current_val + 20
    valid_rows_pool = _gen_valid_rows(k, max_gen, limit=30000)
    rows_by_max = defaultdict(list)
    for row_t, diffs_f in valid_rows_pool:
        rows_by_max[max(row_t)].append((row_t, diffs_f))
    
    no_improve = 0
    max_no_improve = 8000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Alternate between element moves and row swaps
        if random.random() < 0.4:
            # Row-level move
            ri = random.randrange(n_rows)
            old_diffs = current_diffs[ri]
            old_max = row_maxes[ri]
            
            other_diffs = set()
            for j in range(n_rows):
                if j != ri:
                    other_diffs |= current_diffs[j]
            
            found = False
            for tm in range(max(0, old_max - 1), max(0, old_max - 15), -1):
                candidates = rows_by_max.get(tm, [])
                if not candidates:
                    continue
                random.shuffle(candidates)
                for row_t, diffs_f in candidates[:20]:
                    if diffs_f & other_diffs:
                        continue
                    new_row = list(row_t)
                    new_row_maxes = row_maxes[:]
                    new_row_maxes[ri] = tm
                    new_global_max = max(new_row_maxes)
                    
                    if new_global_max < current_val:
                        current_rows[ri] = new_row
                        current_diffs[ri] = set(diffs_f)
                        row_maxes[ri] = tm
                        current_val = new_global_max
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [list(r) for r in current_rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        found = True
                        break
                if found:
                    break
            
            if not found:
                no_improve += 1
        else:
            # Element-level move
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            
            if pi < k:
                max_allowed = current_rows[ri][pi + 1] - 1
            else:
                max_allowed = current_val + 10
            
            # Try decreasing
            for d in range(1, min(20, old_val - min_allowed) + 1):
                new_val = old_val - d
                if new_val < min_allowed:
                    break
                if new_val >= max_allowed and pi < k:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Compute new diffs
                new_diffs = set()
                ok = True
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d_val = cand_row[j] - cand_row[i]
                        if d_val in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d_val)
                    if not ok:
                        break
                
                if not ok:
                    continue
                
                # Check against other rows
                other_diffs = set()
                for j in range(n_rows):
                    if j != ri:
                        other_diffs |= current_diffs[j]
                
                if new_diffs & other_diffs:
                    continue
                
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                new_row_maxes = row_maxes[:]
                new_row_maxes[ri] = cand_row_max
                new_global_max = max(new_row_maxes)
                
                if new_global_max < current_val:
                    current_rows[ri] = cand_row
                    current_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [list(r) for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    break
            else:
                no_improve += 1
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Greedy initialization
    greedy_rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in greedy_rows]
    best_val = max(max(r) for r in greedy_rows)
    
    # 2. Multi-start local search with row swaps
    num_starts = 3
    per_start_budget = (deadline - time.time() - 10.0) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the starting point
        if s == 0:
            start_rows = best_rows
        else:
            # Random perturbation of best_rows
            start_rows = [list(r) for r in best_rows]
            for _ in range(5):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = start_rows[ri][pi]
                min_allowed = start_rows[ri][pi - 1] + 1
                if pi < k:
                    max_allowed = start_rows[ri][pi + 1] - 1
                else:
                    max_allowed = old_val + 10
                if min_allowed <= max_allowed:
                    start_rows[ri][pi] = random.randint(min_allowed, max_allowed)
            # Validate perturbed rows
            if not _is_valid(start_rows):
                start_rows = [list(r) for r in best_rows]
        
        remaining = deadline - time.time() - 5.0
        if remaining < 10.0:
            break
        
        # Run hybrid local search
        result_rows, result_val = _hybrid_local_search(
            start_rows, n_rows, k, min(per_start_budget, remaining)
        )
        
        if result_val < best_val:
            best_val = result_val
            best_rows = [list(r) for r in result_rows]
    
    # 3. Final polish with row-swap local search
    if time.time() < deadline - 3.0:
        polish_budget = min(deadline - time.time() - 3.0, 20.0)
        if polish_budget > 5.0:
            result_rows, result_val = _row_swap_local_search(
                best_rows, n_rows, k, polish_budget
            )
            if result_val < best_val:
                best_val = result_val
                best_rows = [list(r) for r in result_rows]
    
    # Final validity check
    if not _is_valid(best_rows):
        if _is_valid(greedy_rows):
            return [list(r) for r in greedy_rows]
        # Fallback: construct a trivial valid set
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows
    
    return best_rows

# EVOLVE-BLOCK-END
