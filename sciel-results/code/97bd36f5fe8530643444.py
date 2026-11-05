import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # 5s margin for 150s limit
    
    # Number of elements per row
    m = k + 1
    
    # Precompute all pairs (i, j) with 0 <= i < j < m
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    n_pairs = len(pairs)
    
    def get_diffs(row):
        """Get all differences for a row, indexed by pair."""
        return [row[j] - row[i] for i, j in pairs]
    
    def get_diff_set(row):
        """Get set of all positive differences in a row."""
        return {row[j] - row[i] for i in range(m) for j in range(i + 1, m)}
    
    def is_valid_row(row):
        """Check if row is strictly increasing and has unique internal differences."""
        if row[0] != 0:
            return False
        for i in range(1, m):
            if row[i] <= row[i - 1]:
                return False
        diffs = get_diff_set(row)
        return len(diffs) == n_pairs
    
    def validate_all(rows):
        """Check if all rows are valid and have disjoint difference sets."""
        all_diffs = set()
        for row in rows:
            if not is_valid_row(row):
                return False
            dset = get_diff_set(row)
            if all_diffs & dset:
                return False
            all_diffs.update(dset)
        return True
    
    # Strategy: Bounded backtracking search with iterative deepening on scope
    # For each scope S, try to find n_rows rows with max element <= S
    # Use backtracking with constraint propagation
    
    best_rows = None
    best_score = float('inf')
    
    # First, get a quick initial solution with greedy
    def greedy_solve(max_scope=None):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(m - 1):
                last = row[-1]
                found = False
                upper = max_scope if max_scope else 300
                for x in range(last + 1, upper + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    # Get initial solution
    init = greedy_solve()
    if init is not None:
        best_rows = init
        best_score = max(max(r) for r in init)
    
    # Backtracking solver with scope bound
    def backtrack_solve(scope):
        """Try to find n_rows rows with all elements <= scope."""
        if time.time() - start_time > time_limit - 2:
            return None
        
        rows = []
        used_diffs = set()
        
        def bt(row_idx, current_row, pos):
            if time.time() - start_time > time_limit - 2:
                return False
            
            if pos == m:
                # Row complete
                rows.append(list(current_row))
                if row_idx == n_rows - 1:
                    return True
                return bt(row_idx + 1, [0], 1)
            
            if pos == 1:
                # Start of new row
                current_row = [0]
                pos = 1
            
            last = current_row[-1]
            # Try values from last+1 to scope
            # Use some ordering heuristic
            for x in range(last + 1, scope + 1):
                new_diffs = {x - a for a in current_row}
                # Check no internal duplicates
                if len(new_diffs) != pos:
                    continue
                # Check no conflict with used diffs
                if new_diffs & used_diffs:
                    continue
                
                # Tentatively add
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if bt(row_idx, current_row, pos + 1):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start with first row
        result = bt(0, [0], 1)
        if result:
            return rows
        return None
    
    # Local search to improve a solution
    def local_search(rows):
        """Try to reduce the max element by moving individual elements downward."""
        current_max = max(max(r) for r in rows)
        improved = True
        max_iters = 50
        
        for _ in range(max_iters):
            if not improved:
                break
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            
            for i in range(n_rows):
                row = list(rows[i])
                # Try to reduce each element from the end
                for pos in range(m - 1, 0, -1):
                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    
                    if current_val <= min_val:
                        continue
                    
                    # Compute diffs of all other rows
                    other_diffs = set()
                    for j, r in enumerate(rows):
                        if j != i:
                            other_diffs.update(get_diff_set(r))
                    
                    # Try smaller values
                    for v in range(min_val, current_val):
                        new_row = list(row)
                        new_row[pos] = v
                        
                        # Check validity
                        if not is_valid_row(new_row):
                            continue
                        
                        new_diffs = get_diff_set(new_row)
                        if new_diffs & other_diffs:
                            continue
                        
                        # Accept
                        rows[i] = new_row
                        row = new_row
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                        improved = True
                        break  # Move to next position
                if improved:
                    break  # Restart from first row
            
        return rows, current_max
    
    # Main search: iterative deepening on scope
    # Start from best_score and try to go lower
    if best_rows is not None:
        # Try local search on initial solution
        if time.time() - start_time < time_limit - 10:
            ls_rows, ls_score = local_search(best_rows)
            if validate_all(ls_rows) and ls_score < best_score:
                best_score = ls_score
                best_rows = ls_rows
    
    # Iterative deepening: try scopes from best_score-1 down to 105
    current_scope = best_score - 1 if best_score < float('inf') else 200
    
    # Also try randomized restarts with backtracking
    def randomized_backtrack(scope, n_restarts=5):
        """Randomized version of backtracking with different orderings."""
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 3:
                break
            
            rows = []
            used_diffs = set()
            
            def bt_rand(row_idx, current_row, pos):
                if time.time() - start_time > time_limit - 3:
                    return False
                
                if pos == m:
                    rows.append(list(current_row))
                    if row_idx == n_rows - 1:
                        return True
                    return bt_rand(row_idx + 1, [0], 1)
                
                last = current_row[-1]
                # Generate candidate values
                candidates = list(range(last + 1, scope + 1))
                
                # Shuffle with bias toward smaller values
                if restart > 0:
                    # Weighted shuffle
                    weights = [1.0 / (c + 1) ** 0.5 for c in candidates]
                    total_w = sum(weights)
                    shuffled = []
                    remaining = list(zip(candidates, weights))
                    while remaining:
                        r = random.uniform(0, sum(w for _, w in remaining))
                        acc = 0
                        for idx, (c, w) in enumerate(remaining):
                            acc += w
                            if r <= acc:
                                shuffled.append(c)
                                remaining.pop(idx)
                                break
                    candidates = shuffled
                
                for x in candidates:
                    new_diffs = {x - a for a in current_row}
                    if len(new_diffs) != pos:
                        continue
                    if new_diffs & used_diffs:
                        continue
                    
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if bt_rand(row_idx, current_row, pos + 1):
                        return True
                    
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            if bt_rand(0, [0], 1):
                return rows
        
        return None
    
    # Try scopes from high to low
    for scope in range(current_scope, 104, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try deterministic backtracking first
        result = backtrack_solve(scope)
        if result is not None and validate_all(result):
            best_score = scope
            best_rows = result
            # Try local search
            if time.time() - start_time < time_limit - 10:
                ls_rows, ls_score = local_search(best_rows)
                if validate_all(ls_rows) and ls_score < best_score:
                    best_score = ls_score
                    best_rows = ls_rows
            break
        
        # Try randomized backtracking
        if time.time() - start_time < time_limit - 10:
            result = randomized_backtrack(scope, n_restarts=3)
            if result is not None and validate_all(result):
                best_score = scope
                best_rows = result
                if time.time() - start_time < time_limit - 10:
                    ls_rows, ls_score = local_search(best_rows)
                    if validate_all(ls_rows) and ls_score < best_score:
                        best_score = ls_score
                        best_rows = ls_rows
                break
        
        # If we spent too long on this scope, move to next
        if time.time() - start_time > time_limit - 20:
            break
    
    # Final local search if we have time
    if best_rows is not None and time.time() - start_time < time_limit - 3:
        ls_rows, ls_score = local_search(best_rows)
        if validate_all(ls_rows) and ls_score < best_score:
            best_score = ls_score
            best_rows = ls_rows
    
    # Fallback: if no solution found, use greedy
    if best_rows is None:
        best_rows = greedy_solve()
        if best_rows is None:
            # Ultra-simple fallback
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < m:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        row.append(x)
                        used.update(new_diffs)
                    x += 1
                rows.append(row)
            best_rows = rows
    
    return best_rows
