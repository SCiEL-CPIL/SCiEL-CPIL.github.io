"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _diffs_of_row(row):
    """Return the set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _row_scope(row):
    return row[-1]

def _all_diffs(rows):
    used = set()
    for row in rows:
        used |= _diffs_of_row(row)
    return used

def _is_valid(rows):
    used = set()
    for row in rows:
        d = _diffs_of_row(row)
        if d & used:
            return False
        used |= d
    return True

def _total_scope(rows):
    return max(_row_scope(r) for r in rows)

def _greedy_row(used, k):
    """Greedy: build a row of k+1 entries starting at 0, each new mark chosen minimally."""
    row = [0]
    while len(row) < k + 1:
        x = row[-1] + 1
        while True:
            new_diffs = set(x - a for a in row)
            if not (new_diffs & used):
                break
            x += 1
        row.append(x)
        used |= set(x - a for a in row[:-1])
    return row

def _greedy_row_randomized(used, k, window=10):
    """Greedy with a small random window to escape local minima."""
    row = [0]
    while len(row) < k + 1:
        base_x = row[-1] + 1
        found = False
        for _ in range(window * 10):
            x = base_x + random.randint(0, window)
            new_diffs = set(x - a for a in row)
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            # Fallback to deterministic greedy
            x = row[-1] + 1
            while True:
                new_diffs = set(x - a for a in row)
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used |= set(x - a for a in row[:-1])
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for safety
    
    # Phase 1: Deterministic Greedy
    used = set()
    best_rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        best_rows.append(row)
        used |= _diffs_of_row(row)
    
    best_scope = _total_scope(best_rows)
    
    # Phase 2: Randomized Restart Greedy
    # We try to build the entire set of rows from scratch with randomized greedy
    # and keep the best one found.
    while time.time() < deadline:
        new_used = set()
        new_rows = []
        success = True
        for _ in range(n_rows):
            # Use a smaller window for speed, increase if stuck? 
            # For now, a fixed window is faster.
            row = _greedy_row_randomized(new_used, k, window=5)
            new_rows.append(row)
            new_used |= _diffs_of_row(row)
        
        if success:
            new_scope = _total_scope(new_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = new_rows
    
    return best_rows
# EVOLVE-BLOCK-END
