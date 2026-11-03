import time
import random
import sys
import math
from collections import Counter
from itertools import combinations

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

def _backtrack_find_solution(n_rows, k, max_val, deadline):
    """
    Backtracking solver to find a valid set of rows with max entry <= max_val.
    Returns list of rows or None.
    """
    used_diffs = set()
    rows = []
    
    def dfs(row_idx, row_pos, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
            
        if row_pos == k + 1:
            # Finished a row
            rows.append(current_row)
            if dfs(row_idx + 1, 0, [0]):
                return True
            rows.pop()
            return False
            
        # Try to extend current_row
        last_val = current_row[-1]
        
        # Pruning: Can we fit the rest of the row?
        remaining_in_row = k + 1 - row_pos - 1
        if last_val + remaining_in_row > max_val:
            return False
            
        for next_val in range(last_val + 1, max_val + 1):
            # Check if adding next_val causes conflicts
            conflict = False
            new_diffs = []
            for a in current_row:
                d = next_val - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
                
            if conflict:
                continue
                
            # Tentatively add
            current_row.append(next_val)
            used_diffs.update(new_diffs)
            
            if dfs(row_idx, row_pos + 1, current_row):
                return True
                
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False

    return dfs(0, 0, [0])

# ... (keep existing functions _greedy_init, etc.) ...

def construct(n_rows, k):
    deadline = time.time() + 148.0
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 0: Backtracking for small M
    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = total_diffs
    # Try M from lb to lb + 10
    # Limit time per M to 3 seconds
    for M in range(lb, min(lb + 11, 200)):
        if time.time() > deadline - 30:
            break
        m_deadline = time.time() + 3.0
        rows = _backtrack_find_solution(n_rows, k, M, m_deadline)
        if rows:
            if _is_valid(rows, k):
                best_scope = M
                best_rows = rows
                break # Found minimal M, stop
        
    # If backtracking found a solution, return it
    if best_rows is not None:
        return best_rows

    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
    if _is_valid(rows1, k):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]
            
    # Strategy 2: Randomized greedy
    for attempt in range(5):
        if time.time() > deadline - 100:
            break
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.03 + 0.04 * attempt, rng=rng)
        if _is_valid(rows2, k):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]
                
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
        return rows
        
    if not _is_valid(best_rows, k):
        return best_rows
        
    # Run SA optimization
    rows = _sa_optimize(best_rows, n_rows, k, deadline, rng)
    
    if not _is_valid(rows, k):
        return best_rows
    
    if _is_valid(rows, k):
        final_scope = _get_scope(rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = rows
    
    return best_rows
