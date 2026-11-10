"""Difference triangle set (7,5): construct n_rows rows of k+1 increasing ints from 0
with all intra-row positive differences distinct across rows, minimizing the max entry."""
import time
import random
from collections import deque

# Deadline guard
_DEADLINE = 115.0  # seconds (15s safety margin before 130s hard kill)

def _elapsed(start):
    return time.monotonic() - start

def _validate(rows, k):
    """Check that all rows are valid and return the max entry, or None if invalid."""
    used = set()
    max_val = 0
    for row in rows:
        if len(row) != k + 1:
            return None
        if row[0] != 0:
            return None
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return None
                if d in used:
                    return None
                used.add(d)
            max_val = max(max_val, row[i])
    return max_val

def _greedy_row(used, k, start_val=1):
    """Build a single row greedily starting from 0."""
    row = [0]
    x = start_val
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
        x += 1
    return row

def _greedy_all(n_rows, k):
    """Basic greedy construction (same as seed)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
            x += 1
        rows.append(row)
    return rows

def _local_search(rows, k, used, max_iter=20000, deadline=None, start_time=None):
    """
    Local search: try to reduce the max entry by moving individual marks.
    We represent rows as lists and try to decrease the last element first,
    then other elements, checking validity.
    """
    if start_time is None:
        start_time = time.monotonic()
    
    current_max = max(max(row) for row in rows)
    best_rows = [list(r) for r in rows]
    best_max = current_max
    
    # Compute all differences for a row
    def _row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
    
    # Current used set
    cur_used = set()
    for r in best_rows:
        cur_used |= _row_diffs(r)
    
    # Precompute: for each row, which positions can be reduced
    for iteration in range(max_iter):
        if deadline and _elapsed(start_time) > deadline:
            break
        
        improved = False
        # Try to reduce the largest element in each row
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            # Try reducing from the end
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try values from old_val - 1 down to row[pos-1] + 1
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    # Check if new value maintains strict increase
                    if new_val <= row[pos - 1]:
                        break
                    # Compute new diffs for this row
                    new_row = row[:pos] + [new_val] + row[pos+1:]
                    new_row_diffs = _row_diffs(new_row)
                    # Remove old diffs, add new
                    old_row_diffs = _row_diffs(row)
                    removed = old_row_diffs - new_row_diffs
                    added = new_row_diffs - old_row_diffs
                    # Check no conflicts with other rows' diffs
                    other_diffs = cur_used - old_row_diffs
                    if not (added & other_diffs):
                        # Accept
                        cur_used = (cur_used - old_row_diffs) | new_row_diffs
                        row[pos] = new_val
                        new_max = max(max(r) for r in best_rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in best_rows]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_rows, best_max

def _random_restart_greedy(n_rows, k, seed, deadline, start_time):
    """
    Greedy with randomization: shuffle the order of candidate values or
    use a slightly perturbed greedy.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
            x += 1
        rows.append(row)
    return rows

def _beam_search(n_rows, k, beam_width=5, deadline=110.0, start_time=None):
    """
    Beam search over rows: keep multiple partial solutions and expand them.
    State: (used_set, list_of_rows_so_far, current_max)
    """
    if start_time is None:
        start_time = time.monotonic()
    
    # Each beam entry: (used_frozenset, rows_list, max_val)
    # We'll use a list of (used_set, rows) and track max
    beams = [(set(), [])]
    
    for row_idx in range(n_rows):
        if _elapsed(start_time) > deadline:
            break
        new_beams = []
        for used, rows in beams:
            # Generate several candidate rows
            candidates = []
            # Greedy row
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used_copy = used | new_diffs
                x += 1
            candidates.append((row, used))
            
            # Try a few variants: skip some early values
            for skip in range(1, min(4, x - k)):
                row2 = [0]
                x2 = 1 + skip
                valid = True
                while len(row2) < k + 1:
                    new_diffs = {x2 - a for a in row2}
                    if not new_diffs & used:
                        row2.append(x2)
                    x2 += 1
                if len(row2) == k + 1:
                    candidates.append((row2, used))
            
            for row, u in candidates:
                row_diffs = set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
                if not (row_diffs & u):
                    new_used = u | row_diffs
                    new_rows = rows + [row]
                    new_beams.append((new_used, new_rows))
        
        if not new_beams:
            break
        
        # Sort by max entry and keep top beam_width
        new_beams.sort(key=lambda x: max(max(r) for r in x[1]))
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    return beams[0][1]

def _hill_climbing(rows, k, deadline, start_time):
    """
    More aggressive local search: try to reduce any element in any row,
    not just the last one.
    """
    def _row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
    
    cur_used = set()
    for r in rows:
        cur_used |= _row_diffs(r)
    
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    while _elapsed(start_time) < deadline:
        improved = False
        # Try to reduce each position in each row
        for ri in range(len(best_rows)):
            if _elapsed(start_time) > deadline:
                break
            row = best_rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if _elapsed(start_time) > deadline:
                    break
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                # Try to decrease
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    new_diffs = _row_diffs(new_row)
                    old_diffs = _row_diffs(row)
                    added = new_diffs - old_diffs
                    other = cur_used - old_diffs
                    if not (added & other):
                        cur_used = (cur_used - old_diffs) | new_diffs
                        row[pos] = new_val
                        new_max = max(max(r) for r in best_rows)
                        if new_max < best_max:
                            best_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_rows, best_max

def construct(n_rows, k):
    start_time = time.monotonic()
    deadline = start_time + _DEADLINE
    
    # Phase 1: Basic greedy
    rows = _greedy_all(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    # Phase 2: Local search / hill climbing
    if _elapsed(start_time) < deadline - 5:
        rows2, max2 = _hill_climbing(best_rows, k, deadline - 5, start_time)
        if max2 < best_max:
            best_max = max2
            best_rows = rows2
    
    # Phase 3: Beam search with multiple beams
    if _elapsed(start_time) < deadline - 10:
        bs_rows = _beam_search(n_rows, k, beam_width=8, deadline=deadline - 10, start_time=start_time)
        if bs_rows:
            bs_max = max(max(r) for r in bs_rows)
            if bs_max < best_max:
                best_max = bs_max
                best_rows = bs_rows
    
    # Phase 4: More local search on the best
    if _elapsed(start_time) < deadline - 3:
        rows3, max3 = _hill_climbing(best_rows, k, deadline - 3, start_time)
        if max3 < best_max:
            best_max = max3
            best_rows = rows3
    
    # Phase 5: Try a few random restarts if time allows
    if _elapsed(start_time) < deadline - 5:
        for attempt in range(20):
            if _elapsed(start_time) > deadline - 5:
                break
            # Perturbed greedy: start with a random offset for the first mark
            rng = random.Random(attempt * 1000 + 42)
            used = set()
            rows_try = []
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    x += 1
                rows_try.append(row)
            try_max = max(max(r) for r in rows_try)
            if try_max < best_max:
                best_max = try_max
                best_rows = [list(r) for r in rows_try]
            # Quick local search on this
            if _elapsed(start_time) < deadline - 3:
                rows_ls, max_ls = _hill_climbing(best_rows if try_max >= best_max else rows_try, k, deadline - 3, start_time)
                if max_ls < best_max:
                    best_max = max_ls
                    best_rows = rows_ls
    
    # Final validation
    final_max = _validate(best_rows, k)
    if final_max is None:
        # Fallback to pure greedy
        rows = _greedy_all(n_rows, k)
        return rows
    
    return best_rows
