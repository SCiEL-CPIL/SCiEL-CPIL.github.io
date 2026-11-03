"""Difference triangle set (7,5) solver with backtracking + constraint propagation + local search."""
import time
import random
import math
from functools import lru_cache

_DEADLINE = 148.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    return set(_row_diffs(row))

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
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

def _backtrack_row(used, k, max_val, rng, deadline_check=None):
    """
    Backtracking search to build a single row of k+1 entries starting at 0,
    such that all differences are not in 'used'.
    Returns a valid row or None if no solution exists within max_val.
    """
    row = [0]
    
    def _dfs(row, pos, current_max):
        if deadline_check and deadline_check():
            return None
        if len(row) == k + 1:
            return list(row)
        
        # Determine the range of candidates for the next entry
        # The next entry must be > row[-1] and <= max_val
        start_val = row[-1] + 1
        end_val = min(max_val, current_max + 20)  # Limit search width
        
        # Try values in order, with some randomization
        candidates = list(range(start_val, end_val + 1))
        rng.shuffle(candidates)
        
        for candidate in candidates:
            new_diffs = {candidate - a for a in row}
            if new_diffs & used:
                continue
            # Check if this candidate could possibly lead to a valid row
            # (heuristic: not too many conflicts ahead)
            row.append(candidate)
            result = _dfs(row, pos + 1, candidate)
            if result is not None:
                return result
            row.pop()
        
        return None
    
    return _dfs(row, 0, 0)

def _greedy_row(used, k, rng, max_val=None):
    """Fast greedy row construction."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        if rng.random() < 0.03:
            x += rng.randint(1, 3)
            if max_val is not None and x > max_val:
                break
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used = used | new_diffs
    return row if len(row) == k + 1 else None

def _build_initial(n_rows, k, rng, max_scope=None):
    """Build n_rows using a mix of greedy and backtracking."""
    used = set()
    rows = []
    
    for i in range(n_rows):
        row = None
        # Try greedy first
        if max_scope is not None:
            row = _greedy_row(used, k, rng, max_val=max_scope)
        else:
            row = _greedy_row(used, k, rng)
        
        if row is None and max_scope is not None:
            # Try backtracking with tighter bound
            row = _backtrack_row(used, k, max_scope, rng, deadline_check=lambda: _elapsed() > _DEADLINE - 10)
        
        if row is None:
            return None
        
        rows.append(row)
        used |= _row_diffs_set(row)
    
    return rows

def _local_search(rows, k, rng, time_budget=30):
    """
    Local search that tries to reduce the scope by:
    1. Rebuilding individual rows with tighter bounds
    2. Swapping row order to try different assignments
    3. Single-element adjustments
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    
    start_time = time.time()
    temp = 3.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Rebuild one row with a tighter max
            ri = rng.randrange(n_rows)
            old_row = best_rows[ri]
            old_diffs = _row_diffs_set(old_row)
            
            # Compute used diffs excluding this row
            used = set()
            for i, row in enumerate(best_rows):
                if i != ri:
                    used |= _row_diffs_set(row)
            
            # Try to rebuild with a tighter bound
            target_max = min(best_scope, max(old_row) + rng.randint(-5, 2))
            new_row = _backtrack_row(used, k, target_max, rng, deadline_check=lambda: _elapsed() > _DEADLINE - 10)
            
            if new_row is None:
                # Try greedy with the tighter bound
                new_row = _greedy_row(used, k, rng, max_val=target_max)
            
            if new_row is not None:
                new_scope = max(best_scope, max(new_row))
                delta = new_scope - best_scope
                
                if delta <= 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                    best_rows[ri] = new_row
                    if new_scope < best_scope:
                        best_scope = new_scope
        
        elif move_type < 0.7:
            # Try to reduce a single element
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) < 2:
                continue
            
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Try reducing
            for reduction in range(1, 6):
                new_val = old_val - reduction
                if new_val <= 0:
                    break
                if idx > 1 and new_val <= row[idx - 1]:
                    break
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    break
                
                # Check if this change creates conflicts
                old_diffs = _row_diffs_set(row)
                row[idx] = new_val
                new_diffs = _row_diffs_set(row)
                
                # Check conflicts with other rows
                conflict = False
                for i, other_row in enumerate(best_rows):
                    if i == ri:
                        continue
                    other_diffs = _row_diffs_set(other_row)
                    # Only check new diffs that weren't in old_diffs
                    for d in new_diffs - old_diffs:
                        if d in other_diffs:
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    new_scope = _max_entry(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        break
                else:
                    row[idx] = old_val
                    break
            else:
                row[idx] = old_val
        else:
            # Random perturbation: swap two rows
            if n_rows >= 2:
                i, j = rng.sample(range(n_rows), 2)
                best_rows[i], best_rows[j] = best_rows[j], best_rows[i]
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_rows if _is_valid(best_rows, k) else rows

def _refine_scope(rows, k, rng, time_budget=20):
    """
    Iteratively try to reduce the scope by rebuilding rows with progressively
    tighter bounds.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        improved = False
        for ri in range(n_rows):
            if _elapsed() > _DEADLINE - 5:
                break
            
            old_row = best_rows[ri]
            old_diffs = _row_diffs_set(old_row)
            
            # Compute used diffs excluding this row
            used = set()
            for i, row in enumerate(best_rows):
                if i != ri:
                    used |= _row_diffs_set(row)
            
            # Try to find a row with smaller max
            for target_max in range(max(old_row) - 1, 0, -1):
                new_row = _backtrack_row(used, k, target_max, rng, 
                                        deadline_check=lambda: _elapsed() > _DEADLINE - 5)
                if new_row is not None:
                    new_scope = max(best_scope, target_max)
                    if new_scope < best_scope:
                        best_rows[ri] = new_row
                        best_scope = new_scope
                        improved = True
                    break
                elif target_max < max(old_row) - 10:
                    break
        
        if not improved:
            break
    
    return best_rows if _is_valid(best_rows, k) else rows

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Multiple initial constructions with varying strategies
    n_restarts = 20
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 30:
            break
        
        rng2 = random.Random(1000 + i * 997)
        
        # Vary the max_scope to encourage different structures
        if i < 5:
            max_scope = None  # Unconstrained
        else:
            max_scope = 110 + i * 2  # Progressive tightening
        
        rows = _build_initial(n_rows, k, rng2, max_scope=max_scope)
        if rows is None:
            continue
        
        if not _is_valid(rows, k):
            continue
        
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    # Fallback if no valid construction found
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
    
    # Phase 2: Local search to improve
    if _elapsed() < _DEADLINE - 20:
        time_left = _DEADLINE - 20 - _elapsed()
        ls_time = min(40, time_left)
        improved_rows = _local_search(best_rows, k, rng, time_budget=ls_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    # Phase 3: Scope refinement
    if _elapsed() < _DEADLINE - 10:
        time_left = _DEADLINE - 10 - _elapsed()
        refine_time = min(25, time_left)
        if refine_time > 0:
            refined_rows = _refine_scope(best_rows, k, rng, time_budget=refine_time)
            if refined_rows and _is_valid(refined_rows, k):
                new_score = _max_entry(refined_rows)
                if new_score < best_score:
                    best_rows = refined_rows
                    best_score = new_score
    
    # Final validity check
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
