import time
import random
import math
import sys
from collections import Counter

_DEADLINE = 0.0
_BEST = None

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

def _greedy_init(n_rows, k, rng):
    """
    Greedy construction with bitmask difference tracking.
    This provides a valid initial solution to establish an upper bound.
    """
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _rebuild_row(row_idx, rows, k, used_diffs, rng, max_scope):
    """
    Reconstruct a single row to minimize its max element, given the current
    set of used differences from other rows.
    Returns the new row or None if no valid row is found within max_scope.
    """
    # Collect differences used by all OTHER rows
    other_diffs = set()
    for i, r in enumerate(rows):
        if i == row_idx:
            continue
        for d in _row_diffs(r):
            other_diffs.add(d)
    
    # We want to build a row of k+1 integers starting at 0, strictly increasing,
    # with max element <= max_scope, such that all pairwise diffs are NOT in other_diffs.
    
    # Backtracking search for this single row
    # State: current partial row, last element
    # We use a simple recursive backtracking with heuristics.
    
    row = [0]
    last = 0
    
    def backtrack(pos):
        if pos == k + 1:
            return True
        
        # Heuristic: try values that allow the remaining elements to fit
        # If we need (k+1 - pos) more elements, and current last is 'last',
        # the smallest possible next is last+1.
        # The largest possible last element for the whole row is max_scope.
        # So the current element 'x' must be <= max_scope - (k - pos)
        
        # Also, we want to keep the row "tight" to minimize scope, 
        # but we must avoid used diffs.
        
        start = last + 1
        end = max_scope - (k - pos)
        
        if start > end:
            return False
            
        # To find a tight row, we try smaller values first.
        # However, to escape local minima, we might shuffle.
        # For reconstruction, we prefer smaller values to reduce scope.
        
        for x in range(start, end + 1):
            # Check if x creates a conflict with existing row elements
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in other_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if ok:
                row.append(x)
                if backtrack(pos + 1):
                    return True
                row.pop()
                
        return False

    if backtrack(1):
        return row
    return None

def _local_search(rows, k, rng, deadline):
    """
    Iterative row-reconstruction local search.
    Repeatedly pick a random row and try to rebuild it to be tighter.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Precompute used diffs for quick update? 
    # Actually, _rebuild_row recalculates other_diffs each time. 
    # For n=7, k=5, this is O(n * diffs) which is small.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Pick a random row to rebuild
        idx = rng.randint(0, len(current_rows) - 1)
        
        # Try to rebuild with current scope
        # We can try to rebuild with a slightly smaller target scope to force improvement
        # Or just rebuild with current scope and see if it finds a different, potentially tighter solution.
        # Actually, _rebuild_row finds A valid row. It doesn't guarantee it's the tightest.
        # To encourage tightening, we can try to rebuild with target_scope = current_scope.
        # If it finds a row with max < current_scope, we accept.
        
        new_row = _rebuild_row(idx, current_rows, k, None, rng, current_scope)
        
        if new_row is not None:
            new_max = new_row[-1]
            if new_max < current_rows[idx][-1]:
                # Accept
                current_rows[idx] = new_row
                new_scope = _scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                current_scope = new_scope
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart with a different random seed or break
            # Let's try a few more times before giving up on this configuration
            if no_improve_count > max_no_improve * 2:
                break
                
    return best_scope, best_rows

def _construct_csp_solver(n_rows, k, deadline):
    """
    A more sophisticated approach using a simplified CSP solver.
    Since the problem size is small (7 rows, 6 elements), we can use
    a specialized backtracking that orders rows by their "difficulty" (e.g., max element).
    """
    # This is a placeholder for a more complex solver if needed.
    # The local search above is often sufficient for small n.
    pass

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 115.0  # Safety margin
    
    def consider(rows):
        global _BEST
        if rows is None:
            return
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    # Try multiple random seeds for greedy init to find a good starting point
    best_greedy = None
    best_greedy_scope = float('inf')
    
    for seed in range(20):
        r = random.Random(seed)
        rows = _greedy_init(n_rows, k, r)
        s = _scope(rows)
        if s < best_greedy_scope:
            best_greedy_scope = s
            best_greedy = rows
            
    if best_greedy is not None:
        consider(best_greedy)
        
    # Phase 2: Local Search with Row Reconstruction
    if _BEST is not None:
        # Allocate most time to local search
        ls_deadline = _DEADLINE - 5.0
        s, rows = _local_search(_BEST[1], k, rng, ls_deadline)
        consider(rows)
        
    if _BEST is not None:
        return _BEST[1]
    return best_greedy

# Entry point
if __name__ == "__main__":
    # For testing
    rows = construct(7, 5)
    print("Scope:", _scope(rows))
    print("Valid:", _valid(rows, 5))
    for r in rows:
        print(r)
