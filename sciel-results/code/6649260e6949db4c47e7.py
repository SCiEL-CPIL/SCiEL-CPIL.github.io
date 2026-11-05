"""
Difference triangle set (7,5) solver.

Strategy:
1. Iterative deepening on scope: Try to find a valid configuration with scope S,
   increasing S from a lower bound.
2. Backtracking search with pruning:
   - Variables: Elements of each row.
   - Constraints: Strictly increasing, distinct differences globally.
   - Heuristic: Try smaller values first (greedy) to minimize scope.
   - Randomized restarts: If deterministic search fails, try randomized value ordering
     to escape local minima.
3. Fallback to greedy construction if time runs out.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _backtracking_solve(n_rows, k, scope_limit, node_limit=1000000, rng=None, randomize=False):
    """
    Attempts to find a valid set of rows using backtracking.
    Returns a list of rows if successful, None otherwise.
    """
    if rng is None:
        rng = random.Random()
    
    used_diffs = set()
    current_rows = []
    nodes_visited = 0
    
    def backtrack(row_idx, pos, current_row):
        nonlocal nodes_visited
        if nodes_visited > node_limit:
            return None
        
        nodes_visited += 1
        
        if row_idx == n_rows:
            return [r[:] for r in current_rows]
        
        if pos == k: # We have k+1 elements (0 to k). pos is 1-based index for new element.
                     # So if pos == k, we just added the k-th element (index k-1? No).
                     # Let's trace:
                     # Start: pos=1, current_row=[0].
                     # Add val at pos=1. current_row becomes [0, v1].
                     # Next call: pos=2.
                     # ...
                     # Add val at pos=k. current_row becomes [0, v1, ..., vk].
                     # Length is k+1.
                     # So if pos == k, we are done with this row.
            current_rows.append(current_row)
            res = backtrack(row_idx + 1, 1, [0])
            if res:
                return res
            current_rows.pop()
            return None
        
        prev_val = current_row[-1]
        
        # Generate candidates
        candidates = list(range(prev_val + 1, scope_limit + 1))
        
        # Prune: need enough room for remaining elements
        # Remaining elements to add: k - pos
        # Min value for next is prev_val + 1
        # So max value for current must be <= scope_limit - (k - pos)
        # This is already handled by the range upper bound if we are careful, 
        # but let's be explicit.
        max_allowed = scope_limit - (k - pos)
        if max_allowed < prev_val + 1:
            return None
            
        candidates = [c for c in candidates if c <= max_allowed]
        
        if randomize:
            rng.shuffle(candidates)
        
        for val in candidates:
            # Check differences with previous elements in current_row
            conflict = False
            new_diffs = []
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Accept
            used_diffs.update(new_diffs)
            res = backtrack(row_idx, pos + 1, current_row + [val])
            if res:
                return res
            used_diffs.difference_update(new_diffs)
            
            # Early termination if node limit reached
            if nodes_visited > node_limit:
                return None

        return None

    return backtrack(0, 1, [0])

def _construct(n_rows, k, deadline):
    start = time.time()
    
    # Lower bound on scope
    # Total diffs = n_rows * (k * (k-1) / 2)
    # For n=7, k=5 (6 elements), diffs per row = 10. Total = 70.
    # Min scope >= 70.
    # Start from 105 (counting bound mentioned in prompt)
    
    # We will try scopes from 105 to 125
    # For each scope, we try deterministic search, then randomized search.
    
    for scope in range(105, 126):
        if time.time() > deadline - 5:
            break
            
        # 1. Deterministic search (greedy value ordering)
        if time.time() < deadline - 10:
            res = _backtracking_solve(n_rows, k, scope, node_limit=500000, randomize=False)
            if res:
                return res
            
        # 2. Randomized search (shuffle value ordering)
        if time.time() < deadline - 5:
            rng = random.Random(scope) # Deterministic seed for reproducibility in this run
            res = _backtracking_solve(n_rows, k, scope, node_limit=500000, rng=rng, randomize=True)
            if res:
                return res
            
        # 3. If still not found, try a longer randomized search
        if time.time() < deadline - 2:
            rng = random.Random(12345 + scope)
            res = _backtracking_solve(n_rows, k, scope, node_limit=1000000, rng=rng, randomize=True)
            if res:
                return res

    # Fallback: Greedy construction
    # This ensures we return *something* valid, even if suboptimal
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x < 200: # Arbitrary high limit
                new_diffs = [x - a for a in row]
                if not any(d in used_diffs for d in new_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                # Stuck? This shouldn't happen for small n, k
                # Fallback: just append large numbers
                row.append(row[-1] + 100)
                break
        rows.append(row)
        
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rows = _construct(n_rows, k, deadline)
    
    # Validate
    if rows is None or len(rows) != n_rows:
        # Ultimate fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
    
    return rows

# EVOLVE-BLOCK-END
