import time
import random
import math
from collections import Counter

_DEADLINE = 145.0
_START = None

def _row_diffs(row):
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d

def _build_counter(rows):
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c

def _conflicts_from_counter(c):
    return sum(v - 1 for v in c.values() if v > 1)

def _scope(rows):
    return max(r[-1] for r in rows)

def _energy(rows, counter):
    conf = _conflicts_from_counter(counter)
    if conf > 0:
        return 1000 * conf + _scope(rows)
    else:
        return _scope(rows)

def _valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_init(n_rows, k, rng, rand_factor=0.0):
    """
    Generate a set of rows using a randomized greedy approach.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            candidates = []
            start_x = x + 1
            # Look ahead a bit
            for trial_x in range(start_x, start_x + 500):
                new = {trial_x - a for a in row}
                if not (new & used):
                    candidates.append(trial_x)
                    if len(candidates) > 10:
                        break
            if not candidates:
                trial_x = start_x
                while True:
                    new = {trial_x - a for a in row}
                    if not (new & used):
                        candidates.append(trial_x)
                        break
                    trial_x += 1
            
            if rand_factor > 0 and len(candidates) > 1:
                # Pick a random candidate from the top few
                x = rng.choice(candidates[:max(1, int(len(candidates) * rand_factor))])
            else:
                x = candidates[0]
                
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _rebuild_row(used_other, k, best_max, rng, jitter=0.0):
    """
    Rebuild a single row given the differences used by other rows.
    Tries to keep the max value below best_max.
    Returns the row list or None if failed.
    """
    row = [0]
    row_diffs = set()
    x = 0
    # Search limit: try to stay under best_max if possible, but allow a bit of slack
    # Actually, if we want to reduce scope, we should cap at best_max - 1
    # But for kicks, we might allow higher.
    # For now, let's just try to build a valid row.
    # We can pass a max_val parameter to constrain it.
    
    while len(row) < k + 1:
        found = False
        # Search up to best_max
        search_end = best_max
        for trial_x in range(x + 1, search_end + 1):
            new = {trial_x - a for a in row}
            if not (new & used_other) and not (new & row_diffs):
                if jitter > 0 and rng.random() < jitter:
                    continue
                row.append(trial_x)
                row_diffs |= new
                found = True
                break
        if not found:
            return None
        x = row[-1]
    return row

def _local_polish(rows, k, deadline, rng):
    """
    Reduce scope by locally re-optimizing each row.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    improved = True
    pass_count = 0
    max_passes = 10
    
    while improved and time.time() < deadline - 5 and pass_count < max_passes:
        improved = False
        pass_count += 1
        # Process rows in order of decreasing max value
        order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        for i in order:
            if time.time() > deadline - 5:
                break
            
            # Compute used differences from other rows
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= set(_row_diffs(rows[j]))
            
            best_row = list(rows[i])
            best_max = rows[i][-1]
            
            # Try to rebuild with smaller max
            # Start from best_max - 1 and go down? Or just try random rebuilds?
            # Let's try a few rebuilds with decreasing max constraints.
            # Or just try to rebuild with max_val = best_max - 1
            for attempt in range(20):
                jitter = 0.1 * (attempt + 1)
                new_row = _rebuild_row(used_other, k, best_max - 1, rng, jitter=jitter)
                if new_row is not None:
                    best_row = new_row
                    best_max = new_row[-1]
                    # Try to go even lower?
                    # For now, one successful rebuild is an improvement.
                    # We can loop this in the outer while.
                    break
            
            if best_row != rows[i] and best_max < rows[i][-1]:
                rows[i] = best_row
                improved = True
    return rows

def _targeted_descent(rows, k, deadline, rng):
    """
    Directly minimize the global maximum by attempting to reduce the largest elements.
    """
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_energy = _energy(cur_rows, cur_counter)
    
    best_rows = [list(r) for r in cur_rows]
    best_energy = cur_energy
    best_conf = _conflicts_from_counter(cur_counter)
    best_scope = _scope(cur_rows)
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    while time.time() < deadline:
        # Find the row with the maximum last element
        max_idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
        max_row = cur_rows[max_idx]
        
        improved = False
        
        # Try to reduce elements in max_row, starting from the end
        for pos in range(row_len - 1, 0, -1):
            old_val = max_row[pos]
            lo = max_row[pos - 1] + 1
            hi = (max_row[pos + 1] - 1) if pos + 1 < row_len else old_val - 1
            
            if hi < lo:
                continue
                
            # Try values from hi down to lo
            # We want to find a value that is valid and reduces the scope (if pos is last) or at least maintains validity
            
            candidates = []
            if hi - lo < 50:
                candidates = list(range(hi, lo - 1, -1))
            else:
                # Random sample
                for _ in range(10):
                    candidates.append(rng.randint(lo, hi))
                # Always include hi and lo
                candidates.append(hi)
                candidates.append(lo)
                # Sort descending to try to reduce scope
                candidates.sort(reverse=True)
            
            for new_val in candidates:
                if new_val >= old_val:
                    continue
                    
                # Check validity
                old_d = []
                for j in range(row_len):
                    if j != pos:
                        if j < pos:
                            old_d.append(old_val - max_row[j])
                        else:
                            old_d.append(max_row[j] - old_val)
                            
                new_d = []
                for j in range(row_len):
                    if j != pos:
                        if j < pos:
                            new_d.append(new_val - max_row[j])
                        else:
                            new_d.append(max_row[j] - new_val)
                
                new_d_counts = Counter(new_d)
                old_d_counts = Counter(old_d)
                
                valid = True
                for d, cnt in new_d_counts.items():
                    cur_cnt = cur_counter.get(d, 0)
                    rem_cnt = old_d_counts.get(d, 0)
                    net_cnt = cur_cnt - rem_cnt + cnt
                    if net_cnt > 1:
                        valid = False
                        break
                
                if valid:
                    # Apply change
                    for d in old_d:
                        cur_counter[d] -= 1
                        if cur_counter[d] == 0:
                            del cur_counter[d]
                    for d in new_d:
                        cur_counter[d] = cur_counter.get(d, 0) + 1
                        
                    max_row[pos] = new_val
                    
                    new_scope = _scope(cur_rows)
                    new_conf = _conflicts_from_counter(cur_counter)
                    new_energy = _energy(cur_rows, cur_counter)
                    
                    if new_conf == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in cur_rows]
                    
                    improved = True
                    break # Move to next position or restart?
            
            if improved:
                pass
        
        if not improved:
            # If we couldn't improve the max row, try a random move in a random row
            r_idx = rng.randrange(n_rows)
            pos = 1 + rng.randrange(row_len - 1)
            row = cur_rows[r_idx]
            old_val = row[pos]
            lo = row[pos - 1] + 1
            hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 10
            
            if hi > lo:
                new_val = rng.randint(lo, hi)
                if new_val != old_val:
                    old_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                old_d.append(old_val - row[j])
                            else:
                                old_d.append(row[j] - old_val)
                    
                    new_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                new_d.append(new_val - row[j])
                            else:
                                new_d.append(row[j] - new_val)
                    
                    new_d_counts = Counter(new_d)
                    old_d_counts = Counter(old_d)
                    
                    valid = True
                    for d, cnt in new_d_counts.items():
                        cur_cnt = cur_counter.get(d, 0)
                        rem_cnt = old_d_counts.get(d, 0)
                        net_cnt = cur_cnt - rem_cnt + cnt
                        if net_cnt > 1:
                            valid = False
                            break
                    
                    if valid:
                        for d in old_d:
                            cur_counter[d] -= 1
                            if cur_counter[d] == 0:
                                del cur_counter[d]
                        for d in new_d:
                            cur_counter[d] = cur_counter.get(d, 0) + 1
                        row[pos] = new_val
                        
                        new_scope = _scope(cur_rows)
                        new_conf = _conflicts_from_counter(cur_counter)
                        
                        if new_conf == 0 and new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in cur_rows]
                            improved = True

    return best_rows

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # Safety margin
    
    rng = random.Random(12345)
    
    # Phase 1: Generate multiple random greedy starts
    best_start = None
    best_start_energy = float('inf')
    
    for i in range(20):
        if time.time() > deadline - 30:
            break
        rf = 0.05 * (i + 1)
        rows = _greedy_init(n_rows, k, rng, rf)
        counter = _build_counter(rows)
        conf = _conflicts_from_counter(counter)
        scope = _scope(rows)
        energy = 1000 * conf + scope
        
        if energy < best_start_energy:
            best_start_energy = energy
            best_start = [list(r) for r in rows]
            
    if not best_start:
        # Fallback pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        best_start = rows

    # Phase 2: Local Polish (Row Reconstruction)
    # This is the structural idea from the inspiration: rebuilding rows to reduce scope
    polished = _local_polish(best_start, k, deadline, rng)
    
    # Phase 3: Targeted Descent (Refinement Machinery)
    # This is the structural idea from the current program: fine-tuning individual elements
    optimized_rows = _targeted_descent(polished, k, deadline, rng)
    
    # Final Check
    if not _valid(optimized_rows):
        if _valid(polished):
            return polished
        else:
            if _valid(best_start):
                return best_start
            else:
                # Absolute fallback
                used = set()
                rows = []
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                    rows.append(row)
                return rows

    return optimized_rows
