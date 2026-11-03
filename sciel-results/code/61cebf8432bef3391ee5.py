"""Difference triangle set (7,5) solver with backtracking + local search."""
import time
import random
import sys

_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _backtrack_row(used, k, max_val, rng, row_prefix=None):
    """Build a row via backtracking with forward checking.
    Returns a valid row of k+1 entries starting at 0, or None."""
    if row_prefix is None:
        row = [0]
    else:
        row = list(row_prefix)
    
    if len(row) == k + 1:
        return row
    
    last = row[-1]
    # Determine search range for next element
    start = last + 1
    end = min(max_val, last + 20)  # Limit branching width
    
    # Try candidates in a somewhat random order for diversity
    candidates = list(range(start, end + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        # Check new differences against used set
        new_diffs = set()
        ok = True
        for a in row:
            d = x - a
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        # Also check internal consistency (differences among row elements)
        # Since row is strictly increasing, all x-a for a in row are positive
        # and distinct iff x-a values are distinct, which they are since a values are distinct
        
        row.append(x)
        used |= new_diffs
        
        result = _backtrack_row(used, k, max_val, rng, row)
        if result is not None:
            return result
        
        row.pop()
        used -= new_diffs
    
    return None

def _backtrack_full(n_rows, k, max_val, rng):
    """Build all rows via backtracking. Returns list of rows or None."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = _backtrack_row(used, k, max_val, rng)
        if row is None:
            return None
        rows.append(row)
        for d in _row_diffs(row):
            used.add(d)
    
    return rows

def _greedy_row(used, k, rng, max_val=None):
    """Build one row greedily with randomization."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        if rng.random() < 0.1:
            x += rng.randint(1, 3)
            if max_val is not None and x > max_val:
                break
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k, rng, max_val=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _local_search(rows, k, rng, time_budget=30):
    """Aggressive local search to minimize scope via single-entry shifts."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = sum(sum(r) for r in best_rows)
    
    # Build diff counts
    diff_counts = {}
    for row in best_rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    
    row_diff_sets = [set(_row_diffs(r)) for r in best_rows]
    
    start_time = time.time()
    no_improve = 0
    MAX_NO_IMPROVE = 5
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        improved = False
        # Iterate over all rows and positions, try to reduce
        for ri in range(n_rows):
            row = best_rows[ri]
            row_len = len(row)
            for idx in range(1, row_len):
                old_val = row[idx]
                
                # Try reducing by 1 to 15
                for reduction in range(1, 16):
                    new_val = old_val - reduction
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx - 1]:
                        break
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        continue
                    
                    new_row = list(row)
                    new_row[idx] = new_val
                    new_diffs_list = _row_diffs(new_row)
                    new_diff_set = set(new_diffs_list)
                    
                    if len(new_diffs_list) != len(new_diff_set):
                        continue
                    
                    old_diff_set = row_diff_sets[ri]
                    added = new_diff_set - old_diff_set
                    removed = old_diff_set - new_diff_set
                    
                    # Check for conflicts with other rows
                    conflict = False
                    for d in added:
                        other_cnt = diff_counts.get(d, 0)
                        if d in old_diff_set:
                            other_cnt -= 1
                        if other_cnt >= 1:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    new_scope = max(new_row[-1], max((best_rows[i][-1] for i in range(n_rows) if i != ri), default=0))
                    delta_sum = new_val - old_val
                    new_sum = best_sum + delta_sum
                    
                    if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                        # Accept the move
                        for d in removed:
                            cnt = diff_counts.get(d, 0)
                            if cnt == 1:
                                if d in diff_counts:
                                    del diff_counts[d]
                            else:
                                diff_counts[d] = cnt - 1
                        for d in added:
                            old_cnt = diff_counts.get(d, 0)
                            diff_counts[d] = old_cnt + 1
                        
                        row[idx] = new_val
                        row_diff_sets[ri] = new_diff_set
                        best_scope = new_scope
                        best_sum = new_sum
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
            if no_improve >= MAX_NO_IMPROVE:
                break
        else:
            no_improve = 0
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Backtracking search with increasing max_val
    # Start from a reasonable lower bound and increase
    best_rows = None
    best_score = float('inf')
    
    # Lower bound: counting argument gives roughly 105 for (7,5)
    # Each row has C(6,2)=15 differences, 7 rows = 105 total differences
    # So max_val must be at least 105
    start_val = 105
    
    # Try backtracking with increasing max_val
    for max_val in range(start_val, start_val + 30):
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Multiple attempts with different random seeds
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 20:
                break
            rng2 = random.Random(99999 + max_val * 1000 + attempt * 7919)
            rows = _backtrack_full(n_rows, k, max_val, rng2)
            if rows is not None and _is_valid(rows, k):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
                break  # Found valid at this max_val, move to next
    
    # If backtracking didn't find anything, fall back to greedy
    if best_rows is None:
        for i in range(50):
            if _elapsed() > _DEADLINE - 20:
                break
            rng2 = random.Random(12345 + i * 7919)
            rows = _greedy_full(n_rows, k, rng2)
            if rows is not None and _is_valid(rows, k):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
    
    # Final fallback
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    # Phase 2: Local search to compress
    if _elapsed() < _DEADLINE - 8:
        time_left = _DEADLINE - 8 - _elapsed()
        ls_time = min(60, time_left)
        polished = _local_search(best_rows, k, rng, time_budget=ls_time)
        if polished and _is_valid(polished, k):
            new_score = _max_entry(polished)
            if new_score < best_score:
                best_rows = polished
                best_score = new_score
    
    # Final validation
    if not _is_valid(best_rows, k):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
