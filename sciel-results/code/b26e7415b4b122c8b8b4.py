"""Difference-triangle set construction via constraint propagation + randomized search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    """Check if rows form a valid difference triangle set."""
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
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _build_greedy(n_rows, k, rng, deadline, used=None):
    """
    Build a complete solution greedily, trying to keep values small.
    Uses randomized ordering and backtracking-lite (restarts) to escape bad choices.
    Returns (rows, max_val) or (None, None) on failure.
    """
    if used is None:
        used = set()
    
    rows = []
    for r_idx in range(n_rows):
        if time.time() > deadline:
            return None, None
        
        # Try to build one row with small max value
        best_row = None
        best_max = float('inf')
        
        # Multiple attempts to find a good row
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            row = [0]
            row_diffs = set()
            ok = True
            
            for pos in range(k):
                # Determine the range for the next value
                lo = row[-1] + 1
                # We want to keep values small, so bias towards lower values
                # but also ensure we can complete the row
                remaining = k - pos
                # The next value must leave room for (remaining - 1) more values
                # and all differences must be valid
                
                found = False
                # Try values in a randomized order, biased towards smaller
                max_val_try = 200  # upper bound for this row attempt
                for v in range(lo, min(lo + max_val_try, 200)):
                    # Check if adding v is valid
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = v - a
                        if d in row_diffs or d in used or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    
                    # Also check that we can potentially complete the row
                    # (heuristic: not too many small values left)
                    row.append(v)
                    row_diffs.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    ok = False
                    break
            
            if ok and len(row) == k + 1:
                if row[-1] < best_max:
                    best_max = row[-1]
                    best_row = row[:]
                    # If we found a very good row, stop early
                    if best_max <= k * (k + 1) // 2:
                        break
        
        if best_row is None:
            return None, None
        
        rows.append(best_row)
        used.update(_row_diffs(best_row))
    
    return rows, _max_entry(rows)

def _local_search_improve(rows, n_rows, k, rng, deadline):
    """
    Try to improve the solution by rebuilding individual rows with smaller max values.
    Returns (improved_rows, improved_max) or (rows, original_max) if no improvement.
    """
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    used = set()
    for r in cur:
        used.update(_row_diffs(r))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pick the row with the largest max entry
        target = max(range(n_rows), key=lambda i: cur[i][-1])
        old_row = cur[target]
        old_diffs = _row_diffs(old_row)
        used -= old_diffs
        
        # Try to rebuild this row with a smaller max
        best_new = None
        best_new_max = old_row[-1]
        
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            row = [0]
            row_diffs = set()
            ok = True
            
            # Target: try to get max < old_row[-1]
            target_max = old_row[-1] - 1 - attempt // 20
            if target_max < k:
                target_max = old_row[-1] - 1
            
            for pos in range(k):
                lo = row[-1] + 1
                hi = target_max - (k - pos - 1)  # leave room for remaining
                if hi < lo:
                    ok = False
                    break
                
                found = False
                # Randomized search within [lo, hi]
                for _trial in range(50):
                    if _trial < 10:
                        v = lo + _trial  # try small values first
                    else:
                        v = lo + rng.randint(0, max(0, hi - lo))
                    
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = v - a
                        if d in row_diffs or d in used or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    
                    row.append(v)
                    row_diffs.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    ok = False
                    break
            
            if ok and len(row) == k + 1 and row[-1] < best_new_max:
                best_new_max = row[-1]
                best_new = row[:]
        
        if best_new is not None:
            cur[target] = best_new
            used.update(_row_diffs(best_new))
            new_val = _max_entry(cur)
            if new_val < cur_val:
                cur_val = new_val
                improved = True
        else:
            used |= old_diffs
    
    return cur, cur_val

def _simulated_annealing(rows, n_rows, k, rng, deadline):
    """
    Simulated annealing to find better solutions.
    Moves: rebuild a single row, swap two rows, perturb a row.
    Returns the best solution found.
    """
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    used = set()
    for r in cur:
        used.update(_row_diffs(r))
    
    temp = 50.0
    cool_rate = 0.995
    
    while time.time() < deadline and temp > 0.5:
        # Pick a move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Rebuild one row
            target = rng.randint(0, n_rows - 1)
            old_row = cur[target]
            old_diffs = _row_diffs(old_row)
            used -= old_diffs
            
            # Try to build a new row
            new_row = [0]
            row_diffs = set()
            ok = True
            for pos in range(k):
                lo = new_row[-1] + 1
                found = False
                for _trial in range(30):
                    v = lo + rng.randint(0, 30)
                    new_diffs = set()
                    valid = True
                    for a in new_row:
                        d = v - a
                        if d in row_diffs or d in used or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    new_row.append(v)
                    row_diffs.update(new_diffs)
                    found = True
                    break
                if not found:
                    ok = False
                    break
            
            if ok and len(new_row) == k + 1:
                new_val = _max_entry(cur[:target] + [new_row] + cur[target+1:])
                delta = new_val - cur_val
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / temp):
                    cur[target] = new_row
                    used.update(row_diffs)
                    cur_val = new_val
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
            else:
                used |= old_diffs
                
        elif move_type < 0.75:
            # Swap two rows
            i, j = rng.sample(range(n_rows), 2)
            cur[i], cur[j] = cur[j], cur[i]
            # Swap is always valid (same set of differences)
            if _max_entry(cur) < best_val:
                best_val = _max_entry(cur)
                best = [r[:] for r in cur]
            # Swap back if it doesn't help (optional)
            
        else:
            # Perturb: change one value in a row
            target = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # don't change position 0
            old_val = cur[target][pos]
            
            # Try nearby values
            for delta_v in [-1, 1, -2, 2, -3, 3]:
                new_val = old_val + delta_v
                if new_val <= 0 or new_val >= cur[target][pos + 1] if pos < k else new_val > 0:
                    continue
                if pos > 0 and new_val <= cur[target][pos - 1]:
                    continue
                
                # Check validity
                old_diffs = _row_diffs(cur[target])
                new_row = cur[target][:]
                new_row[pos] = new_val
                new_diffs = _row_diffs(new_row)
                
                # Check no conflicts with other rows
                other_diffs = set()
                for i in range(n_rows):
                    if i != target:
                        other_diffs |= _row_diffs(cur[i])
                
                if not (new_diffs & other_diffs) and len(new_diffs) == len(old_diffs):
                    cur[target] = new_row
                    new_max = _max_entry(cur)
                    delta = new_max - cur_val
                    if delta <= 0 or rng.random() < pow(2.718281828, -delta / temp):
                        cur_val = new_max
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r[:] for r in cur]
                    break
        
        temp *= cool_rate
    
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    
    best = None
    best_val = float('inf')
    
    # Phase 1: Multiple greedy constructions
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        rows, val = _build_greedy(n_rows, k, rng, deadline - 10)
        if rows is not None and _valid(rows, n_rows, k):
            if val < best_val:
                best_val = val
                best = [r[:] for r in rows]
    
    # Phase 2: Local search improvement
    if best is not None and time.time() < deadline - 5:
        improved_rows, improved_val = _local_search_improve(best, n_rows, k, rng, deadline - 5)
        if improved_val < best_val:
            best_val = improved_val
            best = improved_rows
    
    # Phase 3: Simulated annealing
    if best is not None and time.time() < deadline - 3:
        sa_rows, sa_val = _simulated_annealing(best, n_rows, k, rng, deadline - 3)
        if sa_val < best_val:
            best_val = sa_val
            best = sa_rows
    
    # Fallback: simple sequential greedy
    if best is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used) and len(nd) == len(row):
                    row.append(x)
                    used |= nd
            rows.append(row)
        if _valid(rows, n_rows, k):
            best = rows
            best_val = _max_entry(best)
    
    return best

# EVOLVE-BLOCK-END
