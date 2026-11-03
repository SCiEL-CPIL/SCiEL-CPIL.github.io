import time
import random
import sys
import math
from collections import Counter
from itertools import combinations
import numpy as np

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0, rng=None):
    """
    Greedy initialization with optional random skipping to break deterministic traps.
    """
    if rng is None:
        rng = random
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and rng.random() < skip_prob and x < 300:
                    skipped = False
                    for extra in range(rng.randint(1, 4)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _backtrack_row(k, diff_counts, max_scope, rng, depth_limit=8):
    """
    Backtracking search to construct a row with minimal max entry.
    Tries to find a row of k+1 elements starting at 0 with all differences
    not in diff_counts, minimizing the last element.
    """
    best_row = None
    best_max = max_scope + 1
    
    def backtrack(current_row, current_max):
        nonlocal best_row, best_max
        
        if len(current_row) == k + 1:
            if current_max < best_max:
                best_max = current_max
                best_row = list(current_row)
            return True
        
        if time.time() > deadline_global - 10:
            return False
            
        # Prune: if current max already exceeds best, stop
        if current_max >= best_max:
            return False
        
        last = current_row[-1]
        # Try candidate values for next element
        # Start from last+1, go up to min(max_scope, best_max - 1)
        start = last + 1
        end = min(max_scope, best_max - 1)
        
        # Add some randomization for diversity
        candidates = list(range(start, end + 1))
        if rng.random() < 0.3 and len(candidates) > 10:
            # Shuffle a subset
            rng.shuffle(candidates[10:])
        
        for t in candidates:
            # Check if all new differences are available
            ok = True
            new_diffs = []
            for a in current_row:
                d = t - a
                if d <= 0:
                    ok = False
                    break
                if diff_counts.get(d, 0) > 0:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            # Recurse
            current_row.append(t)
            # Temporarily mark diffs as used
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            found = backtrack(current_row, max(current_max, t))
            
            # Unmark
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            current_row.pop()
            
            if found and current_max >= best_max:
                return True
        
        return False
    
    backtrack([0], 0)
    return best_row

def _greedy_minimize_row(k, diff_counts, max_scope, rng):
    """
    Greedily construct a row minimizing the max entry.
    At each step, pick the smallest valid next element.
    """
    row = [0]
    used_diffs = set()
    
    for j in range(1, k + 1):
        found = False
        # Try values from current max + 1 upward
        start = row[-1] + 1
        for t in range(start, max_scope + 1):
            ok = True
            new_diffs = []
            for a in row:
                d = t - a
                if d <= 0:
                    ok = False
                    break
                if d in diff_counts and diff_counts[d] > 0:
                    ok = False
                    break
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if ok:
                row.append(t)
                used_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            return None
    
    return row

def _row_rebuild_v2(row, k, diff_counts, rng, max_scope, use_backtrack=True):
    """
    Improved row rebuild: try multiple strategies to find a tighter row.
    """
    best_row = None
    best_max = max_scope + 1
    
    # Strategy 1: Backtracking (if enabled)
    if use_backtrack and max_scope <= 150:
        bt_row = _backtrack_row(k, diff_counts, max_scope, rng)
        if bt_row is not None:
            best_row = bt_row
            best_max = max(bt_row)
            if best_max <= max_scope:
                return best_row
    
    # Strategy 2: Greedy with random restarts
    for attempt in range(20):
        # Perturb the starting point
        if attempt == 0:
            # Pure greedy
            new_row = _greedy_minimize_row(k, diff_counts, max_scope, rng)
        else:
            # Greedy with random jumps
            new_row = [0]
            ok = True
            for j in range(1, k + 1):
                found = False
                start = new_row[-1] + 1
                # Add randomness
                if rng.random() < 0.3:
                    start = new_row[-1] + rng.randint(1, 8)
                
                for t in range(start, min(max_scope + 1, start + 80)):
                    ok_t = True
                    for a in new_row:
                        d = t - a
                        if d <= 0:
                            ok_t = False
                            break
                        if diff_counts.get(d, 0) > 0:
                            ok_t = False
                            break
                    if ok_t:
                        new_row.append(t)
                        found = True
                        break
                
                if not found:
                    ok = False
                    break
            
            if not ok or len(new_row) != k + 1:
                continue
        
        if new_row is not None and len(new_row) == k + 1:
            # Verify all differences are valid
            valid = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if diff_counts.get(d, 0) > 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                cur_max = max(new_row)
                if cur_max < best_max:
                    best_max = cur_max
                    best_row = list(new_row)
                    if cur_max <= max_scope:
                        return best_row
    
    return best_row

def _priority_row_optimization(rows, n_rows, k, deadline, rng):
    """
    Priority-based optimization: always work on the row with the largest entry.
    Rebuild that row to minimize its max, then repeat.
    """
    global deadline_global
    deadline_global = deadline
    
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    no_improve_rounds = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        current_scope = _get_scope(rows)
        if current_scope <= best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in rows]
        
        # Find the row with the largest entry
        max_val = current_scope
        target_row_idx = -1
        for ri in range(n_rows):
            if max(rows[ri]) == max_val:
                target_row_idx = ri
                break
        
        if target_row_idx < 0:
            break
        
        # Build diff counts excluding the target row
        dc = _build_diff_counts(rows)
        old_row = list(rows[target_row_idx])
        _remove_row_diffs(dc, old_row)
        
        # Try to rebuild the target row with a lower max
        new_row = _row_rebuild_v2(old_row, k, dc, rng, max_val - 1, use_backtrack=True)
        
        if new_row is not None and max(new_row) < max_val:
            # Verify validity
            test_rows = [list(r) for r in rows]
            test_rows[target_row_idx] = new_row
            if _is_valid(test_rows, k):
                rows = test_rows
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
        else:
            no_improve_rounds += 1
        
        if no_improve_rounds > max_no_improve:
            break
    
    return best_rows

def _final_scope_reduce(rows, n_rows, k, deadline, rng):
    """Final pass: aggressive scope reduction using backtracking."""
    global deadline_global
    deadline_global = deadline
    
    if not _is_valid(rows, k):
        return rows
    
    rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Try multiple rounds of scope reduction
    for round_num in range(5):
        if time.time() > deadline - 2:
            break
        
        current_scope = _get_scope(rows)
        if current_scope <= 0:
            break
        
        # Find rows that contain the current max
        dc = _build_diff_counts(rows)
        max_val = current_scope
        max_rows = []
        for ri in range(n_rows):
            if max_val in rows[ri]:
                max_rows.append(ri)
        
        if not max_rows:
            break
        
        # Try to rebuild each max-row with a lower max using backtracking
        improved = False
        for ri in max_rows:
            if time.time() > deadline - 2:
                break
            
            old_row = list(rows[ri])
            _remove_row_diffs(dc, old_row)
            new_row = _row_rebuild_v2(old_row, k, dc, rng, max_val - 1, use_backtrack=True)
            
            if new_row is not None and max(new_row) < max_val:
                _add_row_diffs(dc, new_row)
                rows[ri] = new_row
                test_scope = _get_scope(rows)
                
                # Verify validity
                if _is_valid(rows, k):
                    if test_scope < best_scope:
                        best_scope = test_scope
                    improved = True
                    max_val = test_scope
                else:
                    # Revert
                    _remove_row_diffs(dc, new_row)
                    _add_row_diffs(dc, old_row)
                    rows[ri] = old_row
            else:
                _add_row_diffs(dc, old_row)
        
        if not improved:
            break
    
    return rows

def construct(n_rows, k):
    global deadline_global
    deadline = time.time() + 145.0
    deadline_global = deadline
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate multiple diverse initial solutions
    init_configs = [
        (0.0, "pure_greedy"),
        (0.05, "light_skip"),
        (0.10, "medium_skip"),
        (0.15, "heavy_skip"),
        (0.20, "very_heavy_skip"),
    ]
    
    for skip_prob, _ in init_configs:
        if time.time() > deadline - 80:
            break
        rows_init, _ = _greedy_init(n_rows, k, skip_prob=skip_prob, rng=rng)
        if _is_valid(rows_init, k):
            scope_init = _get_scope(rows_init)
            if scope_init < best_scope:
                best_scope = scope_init
                best_rows = [list(r) for r in rows_init]
    
    # Also try some randomized greedy with different seeds
    for seed in [1, 7, 13, 23]:
        if time.time() > deadline - 70:
            break
        rng2 = random.Random(seed)
        rows_init, _ = _greedy_init(n_rows, k, skip_prob=0.08, rng=rng2)
        if _is_valid(rows_init, k):
            scope_init = _get_scope(rows_init)
            if scope_init < best_scope:
                best_scope = scope_init
                best_rows = [list(r) for r in rows_init]
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
        return rows
    
    if not _is_valid(best_rows, k):
        return best_rows
    
    # Phase 2: Priority-based row optimization (replaces SA)
    if time.time() < deadline - 20:
        optimized = _priority_row_optimization(best_rows, n_rows, k, deadline, rng)
        if _is_valid(optimized, k):
            opt_scope = _get_scope(optimized)
            if opt_scope < best_scope:
                best_scope = opt_scope
                best_rows = optimized
    
    # Phase 3: Final aggressive scope reduction with backtracking
    if time.time() < deadline - 3:
        reduced = _final_scope_reduce(best_rows, n_rows, k, deadline, rng)
        if _is_valid(reduced, k):
            red_scope = _get_scope(reduced)
            if red_scope < best_scope:
                best_scope = red_scope
                best_rows = reduced
    
    return best_rows
