"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

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

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _sidon_row(row):
    """Check if a row is a Sidon set (all pairwise differences are distinct)."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _random_sidon_row(k, max_val, used_diffs, rng):
    """Try to build a random Sidon row of k+1 elements starting at 0, within max_val,
    avoiding used_diffs."""
    row = [0]
    used = set(used_diffs)
    for _ in range(k):
        # Try random candidates in increasing order of preference
        candidates = list(range(row[-1] + 1, max_val + 1))
        rng.shuffle(candidates)
        placed = False
        for x in candidates:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # Also check internal Sidon property
                all_diffs = list(new_diffs)
                if len(all_diffs) == len(set(all_diffs)):
                    row.append(x)
                    used |= new_diffs
                    placed = True
                    break
        if not placed:
            return None
    return row

def construct(n_rows, k):
    deadline = time.time() + 110
    random.seed(42)
    rng = random.Random(42)
    
    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    
    # Precompute
    row_diffs = [_row_diffs(r) for r in rows]
    global_diffs = set()
    for s in row_diffs:
        global_diffs |= s
    
    cur = [r[:] for r in rows]
    cur_val = best_val
    
    # Strategy: Iterated Local Search with row-replacement moves
    # Each iteration: pick a row, try to rebuild it with a lower max
    # Use simulated annealing for acceptance
    
    T = 50.0
    T_min = 1.0
    cool = 0.999
    
    # Also maintain a "pressure" on the max entry
    # Focus on rows that contribute to the global max
    
    maxes = [max(r) for r in cur]
    
    # Track consecutive failures for restart
    stuck = 0
    max_stuck = 500
    
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 10000 == 0 and time.time() > deadline - 5:
            break
        
        # Pick a row to modify: bias toward rows with large max entries
        max_max = max(maxes)
        if rng.random() < 0.6:
            # Pick from rows that have the max
            candidates = [i for i in range(n_rows) if maxes[i] == max_max]
            ri = rng.choice(candidates)
        else:
            ri = rng.randrange(n_rows)
        
        old_row = cur[ri]
        old_diffs = row_diffs[ri]
        other_diffs = global_diffs - old_diffs
        
        # Try to rebuild this row with a lower max
        # First, try local perturbations (change one element)
        # Then, if stuck, try full row replacement
        
        improved = False
        
        # Try local moves: change one position
        for attempt in range(3):
            pi = rng.randrange(1, k + 1)
            old_val = old_row[pi]
            min_allowed = old_row[pi - 1] + 1 if pi > 0 else 1
            max_allowed = old_row[pi + 1] - 1 if pi < k else old_val
            
            # Try to decrease
            search_range = min(15, old_val - min_allowed + 1)
            if search_range <= 0:
                continue
            
            candidates = list(range(min_allowed, old_val))
            rng.shuffle(candidates)
            
            for new_val in candidates:
                if new_val > max_allowed:
                    continue
                cand_row = old_row[:]
                cand_row[pi] = new_val
                
                # Check internal Sidon
                if not _sidon_row(cand_row):
                    continue
                
                new_diffs = _row_diffs(cand_row)
                if new_diffs & other_diffs:
                    continue
                
                new_max = max(cand_row)
                new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                
                # Accept if improves or with SA probability
                if new_global_max < cur_val:
                    accept = True
                elif new_global_max == cur_val:
                    accept = rng.random() < 0.3
                else:
                    delta = new_global_max - cur_val
                    accept = rng.random() < math.exp(-delta / max(T, 0.1))
                
                if not accept:
                    continue
                
                # Apply
                global_diffs -= old_diffs
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
            
            if improved:
                break
        
        if not improved:
            # Try full row replacement
            target_max = maxes[ri] - 1
            if target_max < k:
                target_max = k
            
            # Try to build a new row with max <= target_max
            new_row = _random_sidon_row(k, target_max, other_diffs, rng)
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                new_max = max(new_row)
                new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                
                if new_global_max < cur_val:
                    global_diffs -= old_diffs
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
                else:
                    stuck += 1
            else:
                stuck += 1
        
        # Cool down
        T = max(T_min, T * cool)
        
        # If stuck too long, restart from best
        if stuck > max_stuck:
            cur = [r[:] for r in best]
            cur_val = best_val
            row_diffs = [_row_diffs(r) for r in cur]
            global_diffs = set()
            for s in row_diffs:
                global_diffs |= s
            maxes = [max(r) for r in cur]
            T = 50.0
            stuck = 0
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best

# EVOLVE-BLOCK-END
