"""Rewritten program: structural change to constraint-based row construction + ILS."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_row(used, k, max_val, start=0):
    """Build a single row greedily, starting from 'start' value."""
    row = [0]
    cur = 0
    while len(row) < k + 1:
        cur += 1
        if cur > max_val:
            return None
        new_diffs = {cur - a for a in row}
        if new_diffs & used:
            continue
        row.append(cur)
    return row

def _greedy_rows(n_rows, k, max_val):
    """Build all rows greedily, each avoiding differences used by previous rows."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_val)
        if row is None:
            return None
        rows.append(row)
        used |= {row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1)}
    return rows

def _row_diffs(row):
    """Return set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s

def _max_entry(rows):
    return max(max(r) for r in rows)

def _valid(rows, n_rows, k):
    """Check validity: correct size, starts at 0, strictly increasing, all diffs distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, k + 1):
            if row[i] <= row[i - 1]:
                return False
        for d in _row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True

def _try_reduce_entry(rows, n_rows, k, target, time_limit):
    """
    Try to reduce the maximum entry to target using iterative local search.
    Returns best rows found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    
    if best_val <= target:
        return best
    
    # Work with a copy
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_diffs = [_row_diffs(r) for r in cur]
    cur_used = set()
    for s in cur_diffs:
        cur_used |= s
    
    T = 15.0
    T_min = 0.1
    cool = 0.99995
    no_improve = 0
    max_no_improve = 80000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose a row to modify
        ri = random.randrange(n_rows)
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.6:
            # Single position move
            pi = random.randrange(1, k + 1)
            old_val = cur[ri][pi]
            
            # Try to move the position to a smaller value (to reduce max)
            # or a different value
            best_candidate = None
            best_delta = 0
            
            # Try decreasing
            for delta in range(1, 15):
                new_val = old_val - delta
                if new_val <= cur[ri][pi - 1]:
                    break
                cand_row = cur[ri][:]
                cand_row[pi] = new_val
                new_diffs = _row_diffs(cand_row)
                other_diffs = cur_used - cur_diffs[ri]
                if new_diffs & other_diffs:
                    continue
                # Check Sidon within row
                if len(new_diffs) != (k + 1) * k // 2:
                    continue
                cand_max = max(max(r) for i, r in enumerate(cur) if i != ri)
                cand_max = max(cand_max, max(cand_row))
                delta_e = cand_max - cur_val
                if best_candidate is None or delta_e < best_delta:
                    best_candidate = cand_row
                    best_delta = delta_e
            
            # Try increasing (sometimes needed to free up space)
            for delta in range(1, 10):
                new_val = old_val + delta
                if pi < k and new_val >= cur[ri][pi + 1]:
                    break
                if new_val > target + 5:
                    break
                cand_row = cur[ri][:]
                cand_row[pi] = new_val
                new_diffs = _row_diffs(cand_row)
                other_diffs = cur_used - cur_diffs[ri]
                if new_diffs & other_diffs:
                    continue
                if len(new_diffs) != (k + 1) * k // 2:
                    continue
                cand_max = max(max(r) for i, r in enumerate(cur) if i != ri)
                cand_max = max(cand_max, max(cand_row))
                delta_e = cand_max - cur_val
                if best_candidate is None or delta_e < best_delta:
                    best_candidate = cand_row
                    best_delta = delta_e
            
            if best_candidate is None:
                no_improve += 1
                continue
            
            # Accept with SA
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = random.random() < 0.2
            else:
                prob = 1.0 / (1.0 + best_delta / max(T, 0.01))
                accept = random.random() < prob
            
            if accept:
                cur_used -= cur_diffs[ri]
                cur[ri] = best_candidate
                cur_diffs[ri] = _row_diffs(best_candidate)
                cur_used |= cur_diffs[ri]
                cur_val = max(max(r) for r in cur)
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 20000:
                        cur = [r[:] for r in best]
                        cur_val = best_val
                        cur_diffs = [_row_diffs(r) for r in cur]
                        cur_used = set()
                        for s in cur_diffs:
                            cur_used |= s
                        T = 15.0
                        no_improve = 0
        elif move_type < 0.85:
            # Re-solve a row from scratch
            other_diffs = cur_used - cur_diffs[ri]
            # Try to build a new row avoiding other_diffs
            for attempt in range(5):
                row = [0]
                cur_val_row = 0
                success = True
                while len(row) < k + 1:
                    # Try next few values
                    found = False
                    for dv in range(1, 25):
                        candidate = cur_val_row + dv
                        if candidate > target + 10:
                            break
                        new_diffs = {candidate - a for a in row}
                        if new_diffs & other_diffs:
                            continue
                        # Also check within-row Sidon so far
                        all_diffs_so_far = set()
                        for i2 in range(len(row)):
                            for j2 in range(i2 + 1, len(row) + 1):
                                if j2 == len(row):
                                    all_diffs_so_far.add(candidate - row[i2])
                                else:
                                    all_diffs_so_far.add(row[j2] - row[i2])
                        if len(all_diffs_so_far) != len(row) * (len(row) + 1) // 2:
                            continue
                        row.append(candidate)
                        cur_val_row = candidate
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if success and max(row) <= target + 5:
                    new_diffs = _row_diffs(row)
                    if not (new_diffs & other_diffs):
                        cand_max = max(max(r) for i, r in enumerate(cur) if i != ri)
                        cand_max = max(cand_max, max(row))
                        if cand_max < cur_val or (cand_max == cur_val and random.random() < 0.3):
                            cur_used -= cur_diffs[ri]
                            cur[ri] = row
                            cur_diffs[ri] = new_diffs
                            cur_used |= new_diffs
                            cur_val = cand_max
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r[:] for r in cur]
                                no_improve = 0
                            else:
                                no_improve += 1
                            break
            else:
                no_improve += 1
        else:
            # Swap two rows
            if n_rows >= 2:
                rj = random.randrange(n_rows)
                if rj != ri:
                    # Check if swap is valid (it always is since rows are independent)
                    cur[ri], cur[rj] = cur[rj], cur[ri]
                    cur_diffs[ri], cur_diffs[rj] = cur_diffs[rj], cur_diffs[ri]
                    # No change in max or used set
                    no_improve += 1
        
        T = max(T_min, T * cool)
    
    return best

def _binary_search_max(n_rows, k, lower, upper, time_per_try):
    """Binary search for the smallest achievable max entry."""
    best_rows = None
    best_val = upper
    
    # First, get a feasible solution at upper
    rows = _greedy_rows(n_rows, k, upper)
    if rows is None:
        # Greedy failed, try a bit higher
        upper += 10
        rows = _greedy_rows(n_rows, k, upper)
        if rows is None:
            return None
    best_rows = [r[:] for r in rows]
    best_val = _max_entry(best_rows)
    
    # Now try to reduce
    lo = lower
    hi = best_val
    
    while lo < hi and time.time() < time_per_try:
        mid = (lo + hi) // 2
        # Try to achieve mid
        candidate = _try_reduce_entry(best_rows, n_rows, k, mid, time_per_try * 0.3)
        if candidate is not None and _max_entry(candidate) <= mid:
            best_rows = candidate
            best_val = _max_entry(candidate)
            hi = mid
        else:
            lo = mid + 1
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    random.seed(42)
    
    # Estimate bounds
    num_diffs_needed = n_rows * (k + 1) * k // 2
    # Lower bound: need num_diffs_needed distinct differences, each at most M
    # So M >= num_diffs_needed is not quite right; differences range 1..M
    # Actually we need num_diffs_needed distinct values from {1, 2, ..., M}
    # So M >= num_diffs_needed is a very loose lower bound
    # Better: M >= sqrt(2 * num_diffs_needed) approximately
    lower = max(1, int(math.sqrt(2 * num_diffs_needed)) + 1)
    
    # Upper bound: greedy should work with some margin
    upper = int(math.ceil(num_diffs_needed ** 0.6)) + 30
    upper = max(upper, lower + 10)
    
    # Try to find a good solution
    best = _greedy_rows(n_rows, k, upper)
    if best is None:
        # Increase upper bound
        upper += 20
        best = _greedy_rows(n_rows, k, upper)
        if best is None:
            # Fallback: very generous greedy
            upper += 50
            best = _greedy_rows(n_rows, k, upper)
            if best is None:
                # Last resort: simple sequential
                used = set()
                best = []
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                    best.append(row)
    
    best_val = _max_entry(best)
    
    # Now try to improve using local search with time budget
    remaining = deadline - time.time()
    if remaining > 5:
        improved = _try_reduce_entry(best, n_rows, k, best_val - 1, remaining - 5)
        if improved is not None and _max_entry(improved) < best_val:
            best = improved
            best_val = _max_entry(best)
    
    # Final validity check
    if not _valid(best, n_rows, k):
        # Fallback to greedy
        used = set()
        best = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best.append(row)
    
    return best

# EVOLVE-BLOCK-END
