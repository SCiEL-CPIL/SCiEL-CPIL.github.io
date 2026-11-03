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

def _random_row(used, k, max_val):
    """Randomly try to build a row with scope <= max_val."""
    row = [0]
    for _ in range(k):
        # Choose next value randomly in a window
        lo = row[-1] + 1
        hi = min(max_val, lo + 20)
        attempts = 0
        while attempts < 200:
            x = random.randint(lo, hi)
            new_diffs = set(x - a for a in row)
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                break
            attempts += 1
        else:
            return None
    return row

def _local_search(rows, k, budget):
    """Improve rows by trying to reduce scope via random restarts and swaps."""
    best_rows = [list(r) for r in rows]
    best_scope = _total_scope(best_rows)
    start = time.time()
    
    n_rows = len(rows)
    
    # Try to reduce scope by rebuilding individual rows
    while time.time() - start < budget:
        improved = False
        used = _all_diffs(best_rows)
        
        for i in range(n_rows):
            if time.time() - start > budget:
                break
            # Remove row i from used
            row_diffs = _diffs_of_row(best_rows[i])
            used -= row_diffs
            
            # Try to rebuild row i with smaller scope
            current_scope = _row_scope(best_rows[i])
            new_row = None
            
            # Try greedy with constraint
            for target_scope in range(1, current_scope):
                # Try to build a row with last element <= target_scope
                r = [0]
                success = True
                for _ in range(k):
                    lo = r[-1] + 1
                    hi = target_scope
                    if lo > hi:
                        success = False
                        break
                    found = False
                    for x in range(lo, hi + 1):
                        new_diffs = set(x - a for a in r)
                        if not (new_diffs & used):
                            r.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        success = False
                        break
                if success and len(r) == k + 1 and r[-1] < current_scope:
                    new_row = r
                    break
            
            if new_row is not None:
                best_rows[i] = new_row
                used |= _diffs_of_row(new_row)
                new_total = _total_scope(best_rows)
                if new_total < best_scope:
                    best_scope = new_total
                    improved = True
        
        if not improved:
            # Try a full random restart for a few iterations
            for _ in range(10):
                if time.time() - start > budget:
                    break
                new_rows = []
                new_used = set()
                success = True
                for _r in range(n_rows):
                    nr = _random_row(new_used, k, best_scope)
                    if nr is None:
                        success = False
                        break
                    new_rows.append(nr)
                if success:
                    new_total = _total_scope(new_rows)
                    if new_total < best_scope:
                        best_rows = new_rows
                        best_scope = new_total
                        improved = True
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for the 150s kill
    
    # Phase 1: Greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    
    greedy_scope = _total_scope(rows)
    
    # Phase 2: Local search to improve
    remaining = deadline - (time.time() - start_time)
    if remaining > 2.0:
        rows = _local_search(rows, k, remaining - 2.0)
    
    # Phase 3: If still have time, try more random restarts
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        best = rows
        best_scope = _total_scope(best)
        
        for _ in range(100):
            if time.time() - start_time > deadline - 5.0:
                break
            new_rows = []
            new_used = set()
            success = True
            for _r in range(n_rows):
                nr = _random_row(new_used, k, best_scope)
                if nr is None:
                    success = False
                    break
                new_rows.append(nr)
            if success:
                new_scope = _total_scope(new_rows)
                if new_scope < best_scope:
                    best = new_rows
                    best_scope = new_scope
        rows = best
    
    return rows
# EVOLVE-BLOCK-END
