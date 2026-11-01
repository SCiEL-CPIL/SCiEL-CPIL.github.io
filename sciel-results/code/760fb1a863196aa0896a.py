"""Structural rewrite: multi-row coordinated local search with row re-solve."""
import time
import random

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

def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

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
    """Build a single row greedily, avoiding used differences.
    Returns the row or None if no valid row found up to max_val."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None
        # Check if adding x creates any conflict
        ok = True
        for a in row:
            d = x - a
            if d in used:
                ok = False
                break
        if not ok:
            continue
        # Also check internal Sidon property
        # The new diffs are {x - a for a in row}
        # They must not duplicate each other
        new_diffs = [x - a for a in row]
        if len(new_diffs) != len(set(new_diffs)):
            continue
        row.append(x)
        # We don't update used here; caller handles it
    return row

def _greedy_row_with_update(used, k, max_val=None):
    """Build a single row greedily, updating used as we go."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None, used
        new_diffs = [x - a for a in row]
        # Check against used
        conflict = False
        for d in new_diffs:
            if d in used:
                conflict = True
                break
        if conflict:
            continue
        # Check internal uniqueness
        if len(new_diffs) != len(set(new_diffs)):
            continue
        row.append(x)
        for d in new_diffs:
            used.add(d)
    return row, used

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row_with_update(used, k)
        if row is None:
            # Fallback: build without strict limit
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used and len(new) == len(row):
                    row.append(x)
                    used |= new
        rows.append(row)
    return rows

def _random_greedy(n_rows, k, rng):
    """Randomized greedy: at each step, pick from a small window of valid candidates."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Look ahead: find next few valid x values
            valid_xs = []
            for dx in range(1, 30):
                xx = x + dx
                new_diffs = [xx - a for a in row]
                if any(d in used for d in new_diffs):
                    continue
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                valid_xs.append(xx)
                if len(valid_xs) >= 5:
                    break
            if not valid_xs:
                # Expand search
                for dx in range(30, 100):
                    xx = x + dx
                    new_diffs = [xx - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    if len(new_diffs) != len(set(new_diffs)):
                        continue
                    valid_xs.append(xx)
                    if len(valid_xs) >= 5:
                        break
            if not valid_xs:
                # Shouldn't happen, but safety
                xx = x + 1
                while True:
                    new_diffs = [xx - a for a in row]
                    if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                        valid_xs = [xx]
                        break
                    xx += 1
            
            # Pick randomly from valid options, biased toward smaller
            if len(valid_xs) > 1:
                # Weight toward smaller values
                weights = [1.0 / (i + 1) for i in range(len(valid_xs))]
                total = sum(weights)
                r = rng.random() * total
                cum = 0
                chosen = valid_xs[-1]
                for v, w in zip(valid_xs, weights):
                    cum += w
                    if r <= cum:
                        chosen = v
                        break
            else:
                chosen = valid_xs[0]
            
            row.append(chosen)
            for a in row[:-1]:
                used.add(chosen - a)
            x = chosen
        rows.append(row)
    return rows

def _resolve_row(ri, rows, used, k, max_target):
    """Try to rebuild row ri to have max <= max_target, given other rows' used diffs.
    Returns new row or None."""
    other_used = set()
    for i, r in enumerate(rows):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    
    # Greedy rebuild with target
    row = [0]
    x = 0
    while len(row) < k + 1:
        found = False
        while x < max_target:
            x += 1
            new_diffs = [x - a for a in row]
            if any(d in other_used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            found = True
            break
        if not found:
            return None
    return row

def _try_compact_row(ri, rows, used, k):
    """Try to reduce the max entry of row ri by re-solving it."""
    old_row = rows[ri]
    old_max = max(old_row)
    if old_max <= 1:
        return None, rows, used
    
    # Try progressively smaller targets
    for target in range(old_max - 1, max(old_max - 15, 0), -1):
        new_row = _resolve_row(ri, rows, used, k, target)
        if new_row is not None:
            # Update used
            old_diffs = _row_diffs_set(old_row)
            new_diffs = _row_diffs_set(new_row)
            # Remove old, add new
            used -= old_diffs
            used |= new_diffs
            rows[ri] = new_row
            return new_row, rows, used
    return None, rows, used

def _try_swap_diffs(ri, rj, rows, used, k):
    """Try to swap some differences between rows ri and rj to allow compaction.
    This is a heuristic: temporarily remove row ri's diffs, try to rebuild,
    then check if the new diffs conflict with rj."""
    old_i = rows[ri]
    old_j = rows[rj]
    
    other_used = set()
    for idx, r in enumerate(rows):
        if idx != ri and idx != rj:
            other_used |= _row_diffs_set(r)
    
    # Try to rebuild row ri avoiding only other_used (not rj's diffs)
    # Then check if it conflicts with rj
    old_max_i = max(old_i)
    for target in range(old_max_i - 1, max(old_max_i - 10, 0), -1):
        # Greedy rebuild
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            while x < target:
                x += 1
                new_diffs = [x - a for a in row]
                if any(d in other_used for d in new_diffs):
                    continue
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                row.append(x)
                found = True
                break
            if not found:
                break
        if len(row) != k + 1:
            continue
        
        new_diffs_i = _row_diffs_set(row)
        diffs_j = _row_diffs_set(old_j)
        if new_diffs_i & diffs_j:
            continue
        
        # Accept: update
        old_diffs_i = _row_diffs_set(old_i)
        used -= old_diffs_i
        used |= new_diffs_i
        rows[ri] = row
        return True
    
    return False

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    rng = random.Random(42)
    
    # Multiple initializations: greedy + random greedy
    candidates = []
    candidates.append(_build_greedy(n_rows, k))
    for seed in range(5):
        rng2 = random.Random(100 + seed)
        candidates.append(_random_greedy(n_rows, k, rng2))
    
    # Pick best initial
    best = min(candidates, key=_max_entry)
    best_val = _max_entry(best)
    
    # Track used diffs
    used = set()
    for r in best:
        used |= _row_diffs_set(r)
    
    # Local search: row compaction + swap
    cur = [r[:] for r in best]
    cur_used = set(used)
    cur_val = best_val
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Strategy 1: Try to compact the row with the largest max
        row_maxes = [max(r) for r in cur]
        ri = row_maxes.index(max(row_maxes))
        
        new_row, cur, cur_used = _try_compact_row(ri, cur, cur_used, k)
        if new_row is not None:
            new_val = _max_entry(cur)
            if new_val < best_val:
                best_val = new_val
                best = [r[:] for r in cur]
                no_improve = 0
            else:
                no_improve += 1
            continue
        
        # Strategy 2: Try swap between two random rows
        if n_rows >= 2:
            ri = rng.randrange(n_rows)
            rj = rng.randrange(n_rows)
            if ri != rj:
                if _try_swap_diffs(ri, rj, cur, cur_used, k):
                    new_val = _max_entry(cur)
                    if new_val < best_val:
                        best_val = new_val
                        best = [r[:] for r in cur]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
        
        # Strategy 3: Single position perturbation (fallback)
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        moved = False
        for d in range(1, 15):
            for sign in [-1, 1]:
                new_val = old_val + sign * d
                if new_val <= cur[ri][pi - 1]:
                    continue
                if pi < k and new_val >= cur[ri][pi + 1]:
                    continue
                
                cand_row = cur[ri][:]
                cand_row[pi] = new_val
                
                if not _is_sidon(cand_row):
                    continue
                
                new_diffs = _row_diffs_set(cand_row)
                old_diffs = _row_diffs_set(cur[ri])
                other_diffs = cur_used - old_diffs
                
                if new_diffs & other_diffs:
                    continue
                
                # Accept
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = cand_row
                
                new_val = _max_entry(cur)
                if new_val < best_val:
                    best_val = new_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
            if moved:
                break
        
        if not moved:
            no_improve += 1
        
        # Periodic restart
        if no_improve > 10000:
            cur = [r[:] for r in best]
            cur_used = set()
            for r in cur:
                cur_used |= _row_diffs_set(r)
            cur_val = best_val
            no_improve = 0
    
    # Final validation
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
# EVOLVE-BLOCK-END
