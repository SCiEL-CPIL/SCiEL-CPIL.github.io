import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s kill
    
    # Theoretical lower bound on the max element
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Try backtracking with constraint propagation
    # This is more principled than pure greedy
    max_scope_to_try = min(200, lower_bound + 30)
    
    # Try different scope bounds with backtracking
    for scope in range(lower_bound, max_scope_to_try + 1):
        if time.time() > deadline:
            break
            
        result = _backtracking_search(n_rows, k, scope, start_time, deadline)
        if result is not None:
            scope_val = max(max(row) for row in result) if result else 0
            if scope_val < best_scope:
                best_scope = scope_val
                best_rows = result
            # If we found a solution at the lower bound, stop
            if scope_val == lower_bound:
                break
        elif time.time() > deadline:
            break
    
    # Phase 2: If backtracking didn't find good solutions, use randomized greedy as fallback
    if best_rows is None or best_scope > max_scope_to_try:
        seed = 0
        while time.time() < deadline:
            rows = _randomized_greedy(n_rows, k, seed)
            scope = max(max(row) for row in rows) if rows else float('inf')
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope == lower_bound:
                    break
            seed += 1
    
    # If still no solution, use simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
    
    return best_rows

def _backtracking_search(n_rows, k, max_scope, start_time, deadline):
    """
    Backtracking search with constraint propagation.
    Uses a degree heuristic to minimize future constraint violations.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, col_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            # Row complete, move to next row
            rows.append(list(current_row))
            success = backtrack(row_idx + 1, 1, [0])
            if success:
                return True
            rows.pop()
            return False
        
        # Try to place an element at position col_idx in current_row
        # The element must be > current_row[col_idx-1] and <= max_scope
        start_val = current_row[col_idx - 1] + 1
        end_val = min(max_scope, current_row[col_idx - 1] + (max_scope - current_row[col_idx - 1]))
        
        # Collect valid candidates
        valid_candidates = []
        for val in range(start_val, end_val + 1):
            # Check if adding val creates any duplicate differences
            new_diffs = set()
            for i in range(col_idx):
                d = val - current_row[i]
                if d <= 0:
                    continue
                if d in used_diffs:
                    break
                new_diffs.add(d)
            else:
                # All differences are valid
                # Compute "degree": how many future difference slots remain
                # A lower degree means more constraints are satisfied
                # We want to minimize the number of newly used differences that
                # block future possibilities
                valid_candidates.append((val, new_diffs))
        
        if not valid_candidates:
            return False
        
        # Sort by degree heuristic: prefer values that use "less constrained" differences
        # A difference d is "less constrained" if it's larger (more room for future diffs)
        # Or we can prefer values that create differences that are less likely to be needed
        # Simple heuristic: prefer smaller values first (greedy), but with some randomness
        # Better: sort by the number of future rows that would be affected
        
        # For now, use a simple heuristic: try smaller values first
        # But add some randomization to escape local optima
        random.shuffle(valid_candidates)
        
        for val, new_diffs in valid_candidates:
            # Check if this leads to a solution
            used_diffs.update(new_diffs)
            current_row.append(val)
            
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if time.time() > deadline:
                return False
        
        return False
    
    # Start backtracking
    if backtrack(0, 1, [0]):
        return list(rows)
    return None

def _randomized_greedy(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Find valid x's
            valid_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 20 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            if not valid_xs:
                # Fallback: just pick the next integer
                chosen_x = last_x + 1
            else:
                chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            
        rows.append(row)
        
    return rows

def _simple_greedy(n_rows, k):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    
    return rows
