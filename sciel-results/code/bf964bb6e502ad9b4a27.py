"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

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

def _row_diffs_set(row):
    """Return set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

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

def _is_valid_row(row, other_diffs):
    """Check if row is strictly increasing, starts at 0, and has no diff collision with other_diffs."""
    n = len(row)
    if row[0] != 0:
        return False
    for i in range(1, n):
        if row[i] <= row[i - 1]:
            return False
    # Check all pairwise diffs are distinct within row and don't collide with other_diffs
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen or d in other_diffs:
                return False
            seen.add(d)
    return True

def _try_rebuild_row(k, other_diffs, target_max, rng):
    """Try to build a valid row of k+1 elements starting at 0, max <= target_max,
    avoiding other_diffs. Returns row or None."""
    if target_max < k:
        return None
    row = [0]
    for _ in range(k):
        placed = False
        # Try candidates from target_max downward (prefer larger to leave room)
        start = row[-1] + 1
        end = min(target_max, start + 20)
        candidates = list(range(end, start - 1, -1))
        rng.shuffle(candidates)
        for x in candidates:
            # Quick check: new diffs with existing row elements
            new_diffs = {x - a for a in row}
            if new_diffs & other_diffs:
                continue
            # Check internal consistency (all diffs distinct)
            all_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        break
                    all_diffs.add(d)
                else:
                    continue
                break
            else:
                # Check new diffs don't create internal collision
                for d in new_diffs:
                    if d in all_diffs:
                        break
                else:
                    row.append(x)
                    other_diffs = other_diffs | new_diffs
                    placed = True
                    break
        if not placed:
            return None
    return row

def construct(n_rows, k):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Start from greedy
    cur = _build_greedy(n_rows, k)
    best = [r[:] for r in cur]
    best_val = _max_entry(best)
    
    # Precompute per-row diff sets and global diff set
    row_diffs = [_row_diffs_set(r) for r in cur]
    global_diffs = set()
    for s in row_diffs:
        global_diffs |= s
    
    cur_val = best_val
    maxes = [max(r) for r in cur]
    
    # Iterated local search with fast bitmask-style checks
    T = 30.0
    T_min = 0.5
    cool = 0.9995
    stuck = 0
    max_stuck = 200
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a row: bias toward rows with large max entries
        max_max = max(maxes)
        if rng.random() < 0.7:
            candidates = [i for i in range(n_rows) if maxes[i] == max_max]
            ri = rng.choice(candidates)
        else:
            ri = rng.randrange(n_rows)
        
        old_row = cur[ri]
        old_diffs = row_diffs[ri]
        other_diffs = global_diffs - old_diffs
        
        improved = False
        
        # Strategy 1: Try to lower the last element (the max) by rebuilding
        target_max = maxes[ri] - 1
        if target_max >= k:
            # Try multiple rebuilds with randomization
            for attempt in range(5):
                new_row = _try_rebuild_row(k, other_diffs, target_max, rng)
                if new_row is not None:
                    new_max = max(new_row)
                    new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                    
                    if new_global_max < cur_val:
                        # Apply
                        global_diffs -= old_diffs
                        new_diffs = _row_diffs_set(new_row)
                        global_diffs |= new_diffs
                        row_diffs[ri] = new_diffs
                        cur[ri] = new_row
                        maxes[ri] = new_max
                        cur_val = new_global_max
                        
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r[:] for r in cur]
                            stuck = 0
                        else:
                            stuck += 1
                        improved = True
                        break
                    else:
                        # Accept with SA probability for lateral moves
                        if rng.random() < 0.2:
                            global_diffs -= old_diffs
                            new_diffs = _row_diffs_set(new_row)
                            global_diffs |= new_diffs
                            row_diffs[ri] = new_diffs
                            cur[ri] = new_row
                            maxes[ri] = new_max
                            cur_val = new_global_max
                            stuck += 1
                            improved = True
                            break
        
        # Strategy 2: Local move - change one position
        if not improved:
            pi = rng.randrange(1, k + 1)
            old_val = old_row[pi]
            min_allowed = old_row[pi - 1] + 1
            max_allowed = old_row[pi + 1] - 1 if pi < k else old_val
            
            # Try decreasing the value
            search_range = min(10, old_val - min_allowed + 1)
            if search_range > 0:
                candidates = list(range(min_allowed, old_val))
                rng.shuffle(candidates)
                
                for new_val in candidates:
                    if new_val > max_allowed:
                        continue
                    cand_row = old_row[:]
                    cand_row[pi] = new_val
                    
                    # Fast validity check
                    if not _is_valid_row(cand_row, other_diffs):
                        continue
                    
                    new_max = max(cand_row)
                    new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                    
                    if new_global_max < cur_val:
                        accept = True
                    elif new_global_max == cur_val:
                        accept = rng.random() < 0.3
                    else:
                        delta = new_global_max - cur_val
                        accept = rng.random() < math.exp(-delta / max(T, 0.1))
                    
                    if accept:
                        global_diffs -= old_diffs
                        new_diffs = _row_diffs_set(cand_row)
                        global_diffs |= new_diffs
                        row_diffs[ri] = new_diffs
                        cur[ri] = cand_row
                        maxes[ri] = new_max
                        cur_val = new_global_max
                        
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r[:] for r in cur]
                            stuck = 0
                        else:
                            stuck += 1
                        improved = True
                        break
        
        if not improved:
            stuck += 1
        
        # Cool down
        T = max(T_min, T * cool)
        
        # Restart if stuck
        if stuck > max_stuck:
            cur = [r[:] for r in best]
            cur_val = best_val
            row_diffs = [_row_diffs_set(r) for r in cur]
            global_diffs = set()
            for s in row_diffs:
                global_diffs |= s
            maxes = [max(r) for r in cur]
            T = 30.0
            stuck = 0
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best

# EVOLVE-BLOCK-END
