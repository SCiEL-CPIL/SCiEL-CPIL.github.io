"""Difference triangle set (n_rows, k): rows of k+1 increasing ints from 0,
all intra-row positive differences distinct globally; minimize the largest entry."""
import time

_START = time.time()
_DEADLINE = 150.0 - 5.0

def _deadline():
    return time.time() >= _DEADLINE

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _greedy_row(k, used):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row

def construct(n_rows, k):
    # Greedy initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(k, used)
        rows.append(row)
        used |= _row_diffs(row)
    
    rows = [list(r) for r in rows]
    row_diffs = [_row_diffs(r) for r in rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    # Local search: iteratively try to reduce elements
    for iteration in range(200):
        if _deadline():
            break
        improved = False
        for r_idx in range(n_rows - 1, -1, -1):
            if _deadline():
                break
            row = rows[r_idx]
            for j in range(len(row) - 1, 0, -1):
                if _deadline():
                    break
                old_val = row[j]
                cur_diffs = set()
                for i in range(j):
                    cur_diffs.add(row[j] - row[i])
                for k2 in range(j + 1, len(row)):
                    cur_diffs.add(row[k2] - row[j])
                
                other_diffs = all_diffs - cur_diffs
                
                for new_val in range(row[j-1] + 1, old_val):
                    new_diffs = set()
                    for i in range(j):
                        new_diffs.add(new_val - row[i])
                    for k2 in range(j + 1, len(row)):
                        new_diffs.add(row[k2] - new_val)
                    
                    if new_diffs & other_diffs:
                        continue
                    if len(new_diffs) != len(cur_diffs):
                        continue
                    
                    row[j] = new_val
                    all_diffs = (all_diffs - cur_diffs) | new_diffs
                    row_diffs[r_idx] = _row_diffs(row)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return rows
