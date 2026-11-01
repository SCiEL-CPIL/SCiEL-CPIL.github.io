"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.
Uses a targeted local search with efficient difference tracking, multiple restart strategies,
and a final polish pass.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    s = set()
    for row in rows:
        s.update(_row_diffs(row))
    return s

def _is_valid(rows, k):
    """Check validity of the rows."""
    if len(rows) == 0:
        return True
    expected_len = k + 1
    for row in rows:
        if len(row) != expected_len:
            return False
        if row[0] != 0:
            return False
        for i in range(1, expected_len):
            if row[i] <= row[i - 1]:
                return False
    diffs = _all_diffs(rows)
    expected_count = len(rows) * (k * (k + 1) // 2)
    return len(diffs) == expected_count

def _greedy_init(n_rows, k):
    """Greedy initialization: build rows one by one, each starting at 0."""
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

def _interleaved_greedy(n_rows, k):
    """Interleaved greedy: build all rows simultaneously, adding one element at a time."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    for _ in range(k):
        # For each row, find the smallest next value
        next_vals = []
        for i in range(n_rows):
            row = rows[i]
            x = row[-1]
            while True:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    next_vals.append((x, i, new))
                    break
        # Sort by value and add in order
        next_vals.sort()
        for x, i, new in next_vals:
            rows[i].append(x)
            used |= new
    return rows

def _round_robin_greedy(n_rows, k):
    """Round-robin greedy: alternate between rows, each time adding the next smallest valid element."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    for _ in range(k):
        for i in range(n_rows):
            row = rows[i]
            x = row[-1]
            while True:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                    break
    return rows

def _max_entry(rows):
    return max(max(r) for r in rows)

def _local_search(rows, k, time_limit, rng, verbose=False):
    """Local search to reduce max entry by moving elements within rows."""
    best_rows = [r[:] for r in rows]
    best_score = _max_entry(best_rows)
    
    # Build used set
    used = _all_diffs(rows)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 100000
    n = len(rows)
    k1 = k + 1
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row and position
        row_idx = rng.randint(0, n - 1)
        pos = rng.randint(1, k1 - 1)
        row = rows[row_idx]
        old_val = row[pos]
        
        # Try to move downward
        lo = row[pos - 1] + 1
        hi = old_val - 1
        if lo < hi:
            # Try a few random candidates
            for _ in range(3):
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                
                # Check ordering
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < k1 - 1 and new_val >= row[pos + 1]:
                    continue
                
                # Try the move
                success = True
                # Remove old differences
                temp_used = set(used)
                for i in range(k1):
                    if i == pos:
                        continue
                    d = abs(old_val - row[i])
                    temp_used.discard(d)
                
                # Check new differences
                for i in range(k1):
                    if i == pos:
                        continue
                    d = abs(new_val - row[i])
                    if d in temp_used:
                        success = False
                        break
                
                if success:
                    # Apply
                    row[pos] = new_val
                    for i in range(k1):
                        if i != pos:
                            d = abs(new_val - row[i])
                            temp_used.add(d)
                    used = temp_used
                    no_improve = 0
                    
                    current_max = _max_entry(rows)
                    if current_max < best_score:
                        best_score = current_max
                        best_rows = [r[:] for r in rows]
                    break
                else:
                    no_improve += 1
        else:
            no_improve += 1
    
    return best_rows, best_score

def _polish(rows, k, time_limit, rng):
    """Final polish: try to reduce max entry by targeting the row with the largest entry."""
    best_rows = [r[:] for r in rows]
    best_score = _max_entry(best_rows)
    used = _all_diffs(rows)
    
    deadline = time.time() + time_limit
    n = len(rows)
    k1 = k + 1
    
    # Find the row with the largest entry
    max_row_idx = max(range(n), key=lambda i: rows[i][-1])
    
    # Focus on the last few positions of the max row
    for pos in range(k1 - 1, max(0, k1 - 4), -1):
        row = rows[max_row_idx]
        old_val = row[pos]
        lo = row[pos - 1] + 1
        hi = old_val - 1
        
        if lo < hi:
            # Try all values from hi down to lo
            for new_val in range(hi, lo - 1, -1):
                if time.time() > deadline:
                    break
                
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < k1 - 1 and new_val >= row[pos + 1]:
                    continue
                
                # Try the move
                success = True
                temp_used = set(used)
                for i in range(k1):
                    if i == pos:
                        continue
                    d = abs(old_val - row[i])
                    temp_used.discard(d)
                
                for i in range(k1):
                    if i == pos:
                        continue
                    d = abs(new_val - row[i])
                    if d in temp_used:
                        success = False
                        break
                
                if success:
                    row[pos] = new_val
                    for i in range(k1):
                        if i != pos:
                            d = abs(new_val - row[i])
                            temp_used.add(d)
                    used = temp_used
                    
                    current_max = _max_entry(rows)
                    if current_max < best_score:
                        best_score = current_max
                        best_rows = [r[:] for r in rows]
                    break
    
    return best_rows, best_score

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Phase 1: Try multiple initialization strategies
    init_strategies = [
        _greedy_init,
        _interleaved_greedy,
        _round_robin_greedy,
    ]
    
    best_rows = None
    best_score = float('inf')
    
    for init_fn in init_strategies:
        if time.time() > deadline - 10:
            break
        rows = init_fn(n_rows, k)
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k)
        best_score = _max_entry(best_rows)
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Phase 2: Local search on the best initialization
    rng = random.Random(42)
    ls_time = min(40, deadline - time.time() - 5)
    if ls_time > 5:
        rows, score = _local_search(best_rows, k, ls_time, rng)
        if score < best_score:
            best_score = score
            best_rows = rows
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Phase 3: Random restarts with local search
    remaining = deadline - time.time()
    if remaining > 10:
        attempts = 0
        while time.time() < deadline - 3 and attempts < 15:
            seed = 1000 + attempts
            rng2 = random.Random(seed)
            
            # Randomized greedy
            used = set()
            rows2 = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Try a few random candidates
                    found = False
                    for attempt in range(5):
                        if attempt == 0:
                            cand = x + 1
                        else:
                            cand = x + 1 + rng2.randint(0, 15)
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            found = True
                            break
                    if not found:
                        x += 1
                        while True:
                            new = {x - a for a in row}
                            if not new & used:
                                row.append(x)
                                used |= new
                                break
                            x += 1
                rows2.append(row)
            
            score2 = _max_entry(rows2)
            if score2 < best_score:
                best_score = score2
                best_rows = [r[:] for r in rows2]
            
            # Quick local search
            rem = deadline - time.time()
            if rem > 5 and score2 <= best_score + 5:
                rows2, score2 = _local_search(rows2, k, min(5, rem - 1), rng2)
                if score2 < best_score:
                    best_score = score2
                    best_rows = rows2
            
            attempts += 1
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        polish_time = min(10, deadline - time.time() - 2)
        if polish_time > 2:
            rng3 = random.Random(12345)
            best_rows, best_score = _polish(best_rows, k, polish_time, rng3)
    
    return best_rows

# EVOLVE-BLOCK-END
