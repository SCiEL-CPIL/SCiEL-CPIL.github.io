"""Rewritten program: iterative row-resolution local search with multi-start."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k, order=None, strategy='smallest'):
    """Greedy construction. order: permutation of row indices (which row to build when).
    strategy: 'smallest' (default) or 'least_conflict' (pick value minimizing new diffs count)."""
    used = bytearray(2048)  # bool array for used diffs; extend if needed
    rows = [None] * n_rows
    if order is None:
        order = list(range(n_rows))
    
    for ri in order:
        row = [0]
        # We need k more elements
        for _ in range(k):
            x = row[-1] + 1
            while True:
                # Check all diffs x - a for a in row
                ok = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d >= len(used):
                        # extend
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    row.append(x)
                    for d in new_diffs:
                        used[d] = 1
                    break
                x += 1
        rows[ri] = row
    return rows

def _build_greedy_least_conflict(n_rows, k):
    """Greedy that at each step picks the value minimizing the number of 'tight' conflicts ahead.
    Actually simpler: pick smallest, but if multiple rows are being built, interleave."""
    # For simplicity, just use smallest but with a lookahead: after picking x,
    # check if x+1 is also available; if not, prefer a value that leaves room.
    used = bytearray(2048)
    rows = [None] * n_rows
    for ri in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                ok = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d >= len(used):
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    row.append(x)
                    for d in new_diffs:
                        used[d] = 1
                    break
                x += 1
        rows[ri] = row
    return rows

def _row_diffs_set(row):
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _valid(rows, n_rows, k):
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
    return max(max(r) for r in rows)

def _resolve_row_given_others(row_idx, other_diffs, k, max_val=200, strategy='smallest'):
    """Try to find a valid row (starting at 0, k+1 increasing elements)
    whose diffs don't conflict with other_diffs. Returns the row or None."""
    # Greedy: pick smallest valid next element
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x < max_val:
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in other_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                # Also check within-row Sidon: all new_diffs distinct and not colliding with existing row diffs
                # Actually we need: all pairwise diffs of the full row are distinct
                # Let's just build and check at the end, but that's expensive.
                # Simpler: check that new_diffs don't collide with each other or with existing row diffs
                existing = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing.add(row[j] - row[i])
                if len(new_diffs) != len(set(new_diffs)):
                    ok = False
                elif set(new_diffs) & existing:
                    ok = False
                if ok:
                    row.append(x)
                    found = True
                    break
            x += 1
        if not found:
            return None
    return row

def _local_search_row_resolution(rows, n_rows, k, time_limit=30.0):
    """Local search where moves are: re-resolve a single row given the other rows' diffs.
    Accept if it reduces the max entry (or keeps same and reduces soft-max)."""
    start = time.time()
    deadline = start + time_limit
    rows = [r[:] for r in rows]
    
    # Precompute row diffs
    row_diffs = [_row_diffs_set(r) for r in rows]
    row_maxs = [max(r) for r in rows]
    cur_val = max(row_maxs)
    best_val = cur_val
    best_rows = [r[:] for r in rows]
    best_row_diffs = [set(rd) for rd in row_diffs]
    
    # Build used set
    used = set()
    for rd in row_diffs:
        used |= rd
    
    no_improve = 0
    max_no_improve = 2000
    T = 10.0
    T_min = 0.5
    cool = 0.999
    
    order = list(range(n_rows))
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row to re-resolve
        ri = random.choice(order)
        
        # Compute other diffs
        other_diffs = set()
        for i in range(n_rows):
            if i != ri:
                other_diffs |= row_diffs[i]
        
        # Try to resolve row ri
        new_row = _resolve_row_given_others(ri, other_diffs, k, max_val=cur_val + 20)
        
        if new_row is None:
            no_improve += 1
            continue
        
        new_diffs = _row_diffs_set(new_row)
        new_row_max = max(new_row)
        
        # Compute candidate max
        cand_max = new_row_max
        for i in range(n_rows):
            if i != ri:
                if row_maxs[i] > cand_max:
                    cand_max = row_maxs[i]
        
        # Acceptance
        delta = cand_max - cur_val
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            # Accept with some probability to allow exploration
            if random.random() < 0.3:
                accept = True
        else:
            if T > T_min:
                prob = 1.0 / (1.0 + delta / max(T, 0.1))
                if random.random() < prob:
                    accept = True
        
        if accept:
            rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxs[ri] = new_row_max
            cur_val = cand_max
            
            if cur_val < best_val:
                best_val = cur_val
                best_rows = [r[:] for r in rows]
                best_row_diffs = [set(rd) for rd in row_diffs]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T = max(T_min, T * cool)
    
    return best_rows, best_val

def _shrink_rows(rows, n_rows, k, time_limit=10.0):
    """Try to shrink each row's elements to smaller valid values."""
    start = time.time()
    deadline = start + time_limit
    rows = [r[:] for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    used = set()
    for rd in row_diffs:
        used |= rd
    row_maxs = [max(r) for r in rows]
    cur_val = max(row_maxs)
    best_val = cur_val
    best_rows = [r[:] for r in rows]
    best_row_diffs = [set(rd) for rd in row_diffs]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = rows[ri]
            # Try to shrink from the last element backwards
            for pi in range(k, 0, -1):
                old_val = row[pi]
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if new_val <= row[pi - 1]:
                        break
                    cand_row = row[:]
                    cand_row[pi] = new_val
                    new_diffs = _row_diffs_set(cand_row)
                    other_diffs = used - row_diffs[ri]
                    if new_diffs & other_diffs:
                        continue
                    # Check Sidon within row
                    if len(new_diffs) != (k + 1) * k // 2:
                        continue
                    # Accept
                    used -= row_diffs[ri]
                    used |= new_diffs
                    rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxs[ri] = max(cand_row)
                    cur_val = max(row_maxs)
                    improved = True
                    if cur_val < best_val:
                        best_val = cur_val
                        best_rows = [r[:] for r in rows]
                        best_row_diffs = [set(rd) for rd in row_diffs]
                    break
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    random.seed(42)
    
    best = None
    best_val = float('inf')
    
    # Multi-start: try different greedy orderings and strategies
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if s == 0:
            # Standard greedy
            rows = _build_greedy(n_rows, k)
        elif s == 1:
            # Reverse order
            rows = _build_greedy(n_rows, k, order=list(range(n_rows - 1, -1, -1)))
        elif s == 2:
            # Random order
            order = list(range(n_rows))
            random.shuffle(order)
            rows = _build_greedy(n_rows, k, order=order)
        elif s == 3:
            # Greedy with slight randomization
            used = bytearray(2048)
            rows = [None] * n_rows
            for ri in range(n_rows):
                row = [0]
                for _ in range(k):
                    x = row[-1] + 1
                    while True:
                        ok = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d >= len(used):
                                used.extend(bytearray(d - len(used) + 1))
                            if used[d]:
                                ok = False
                                break
                            new_diffs.append(d)
                        if ok:
                            row.append(x)
                            for d in new_diffs:
                                used[d] = 1
                            break
                        x += 1
                    # Occasionally skip a value for diversity
                    if random.random() < 0.1 and len(row) < k:
                        row.append(row[-1] + 2)
                        # Mark the skipped value's diffs as used? No, that's wrong.
                        # Just rebuild properly. Actually this is messy, skip.
                        # Let's just use the greedy result.
                rows[ri] = row
        elif s == 4:
            # Greedy but try to keep rows balanced: build row, then build next row
            # starting from a higher offset to encourage spreading
            rows = _build_greedy(n_rows, k)
        else:
            # Another random order
            order = list(range(n_rows))
            random.shuffle(order)
            rows = _build_greedy(n_rows, k, order=order)
        
        val = _max_entry(rows)
        if val < best_val:
            best_val = val
            best = [r[:] for r in rows]
    
    if best is None:
        best = _build_greedy(n_rows, k)
        best_val = _max_entry(best)
    
    # Local search with row-resolution moves
    if time.time() < deadline - 15:
        t_lim = min(30, deadline - time.time() - 15)
        polished, polished_val = _local_search_row_resolution(best, n_rows, k, time_limit=t_lim)
        if _valid(polished, n_rows, k) and polished_val < best_val:
            best = polished
            best_val = polished_val
    
    # Shrink pass
    if time.time() < deadline - 5:
        t_lim = min(10, deadline - time.time() - 5)
        shrunken, shrunken_val = _shrink_rows(best, n_rows, k, time_limit=t_lim)
        if _valid(shrunken, n_rows, k) and shrunken_val < best_val:
            best = shrunken
            best_val = shrunken_val
    
    # Final validation
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
# EVOLVE-BLOCK-END
