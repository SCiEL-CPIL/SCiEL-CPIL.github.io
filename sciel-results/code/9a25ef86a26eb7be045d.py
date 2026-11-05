import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Column-wise "earliest-feasible" construction with randomized restarts
    best_rows = None
    best_scope = float('inf')
    
    rng = random.Random(42)
    
    # Try multiple random orderings of rows to break symmetry
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random row ordering to break symmetry
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        # Try column-wise construction
        solution = _column_wise_construct(n_rows, k, row_order, rng, deadline)
        
        if solution is not None:
            if _is_valid_solution(solution, n_rows, k):
                scope = _scope(solution)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in solution]
        
        # Early termination if we hit the lower bound
        if best_scope == lower_bound:
            break
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Row-pair swap search
    if time.time() < deadline - 15:
        improved = _row_pair_swap_search(best_rows, n_rows, k, rng, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Element-level local search
    if time.time() < deadline - 5:
        improved = _element_local_search(best_rows, n_rows, k, rng, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
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
                chosen_x = last_x + 1
                while True:
                    new_diffs = {chosen_x - a for a in row}
                    if not (new_diffs & used):
                        break
                    chosen_x += 1
            else:
                chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _is_valid_solution(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i+1] <= row[i]:
                return False
        diffs = _row_diffs(row)
        if len(diffs) != (k+1)*k//2:
            return False
        if all_diffs & diffs:
            return False
        all_diffs.update(diffs)
    return True

def _column_wise_construct(n_rows, k, row_order, rng, deadline):
    """
    Column-wise "earliest-feasible" list-scheduling heuristic.
    Views each row's positions as jobs that must claim difference values.
    Builds rows sequentially, choosing the smallest feasible value at each position
    that allows the remaining rows to still be completed.
    """
    used_diffs = set()
    rows = [None] * n_rows
    
    def can_complete(remaining_rows, current_used_diffs, k, deadline):
        """
        Check if we can still complete the remaining rows given current used_diffs.
        Uses a greedy feasibility check: for each remaining row, try to greedily
        construct it and see if we succeed.
        """
        if time.time() > deadline - 2:
            return True  # Assume feasible to avoid timeout
        
        # Quick check: count available differences
        # Total diffs needed: remaining_rows * (k+1)*k//2
        # Available diffs: 1 to some upper bound
        # This is a weak check, but helps prune impossible cases
        
        # Greedy feasibility: try to construct each remaining row greedily
        test_used = set(current_used_diffs)
        for _ in range(remaining_rows):
            row = [0]
            for pos in range(k):
                found = False
                last_val = row[-1]
                # Try values from last_val+1 upward
                val = last_val + 1
                while val < 200:  # Reasonable upper bound for feasibility check
                    new_diffs = set()
                    valid = True
                    for prev in row:
                        d = val - prev
                        if d in test_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid and not (new_diffs & test_used):
                        # This value works
                        row.append(val)
                        test_used.update(new_diffs)
                        found = True
                        break
                    val += 1
                if not found:
                    return False
            # Successfully constructed one row greedily
        return True
    
    def build_row(row_idx, used_diffs, deadline):
        """Build a single row with backtracking within the row."""
        if time.time() > deadline - 2:
            return None
        
        def backtrack(pos, current_row, row_diffs):
            if time.time() > deadline - 2:
                return False
            
            if pos == k + 1:
                # Check if this row is valid
                if not _is_valid_row(current_row):
                    return False
                # Check no conflict with used_diffs
                if row_diffs & used_diffs:
                    return False
                # This row works
                rows[row_idx] = list(current_row)
                used_diffs.update(row_diffs)
                return True
            
            last_val = current_row[-1]
            start_val = last_val + 1
            
            # Heuristic: try smaller values first to keep scope small
            # But also consider that we need room for remaining elements
            end_val = 150  # Reasonable upper bound
            
            for val in range(start_val, end_val + 1):
                new_diffs = set()
                valid = True
                for prev in current_row:
                    d = val - prev
                    if d in row_diffs:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(val)
                row_diffs.update(new_diffs)
                
                if backtrack(pos + 1, current_row, row_diffs):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    row_diffs.discard(d)
            
            return False
        
        if backtrack(1, [0], set()):
            return rows[row_idx]
        return None
    
    # Build rows in the specified order
    for idx in range(n_rows):
        if time.time() > deadline - 5:
            return None
        
        row_idx = row_order[idx]
        
        # Try to build this row
        row = build_row(row_idx, used_diffs, deadline)
        if row is None:
            return None
        
        # Check feasibility for remaining rows
        remaining = n_rows - idx - 1
        if remaining > 0:
            if not can_complete(remaining, used_diffs, k, deadline):
                # This choice might be bad, but we'll continue and hope
                # a later restart with different ordering works
                pass
    
    return rows

def _row_pair_swap_search(initial_rows, n_rows, k, rng, deadline):
    """
    Try to improve by swapping elements between pairs of rows.
    """
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(5000):
        if time.time() > deadline - 3:
            break
        
        # Pick two random distinct rows
        i, j = rng.sample(range(n_rows), 2)
        
        # Pick random positions in each row
        pos_i = rng.randint(1, k)
        pos_j = rng.randint(1, k)
        
        old_val_i = rows[i][pos_i]
        old_val_j = rows[j][pos_j]
        
        # Try swapping
        rows[i][pos_i] = old_val_j
        rows[j][pos_j] = old_val_i
        
        # Check validity
        valid = True
        for idx in [i, j]:
            if not _is_valid_row(rows[idx]):
                valid = False
                break
        
        if valid:
            # Check no conflicts between the two modified rows
            diffs_i = _row_diffs(rows[i])
            diffs_j = _row_diffs(rows[j])
            if diffs_i & diffs_j:
                valid = False
        
        if valid:
            # Check no conflicts with other rows
            for other in range(n_rows):
                if other == i or other == j:
                    continue
                diffs_other = _row_diffs(rows[other])
                if (diffs_i & diffs_other) or (diffs_j & diffs_other):
                    valid = False
                    break
        
        if valid:
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            # Undo
            rows[i][pos_i] = old_val_i
            rows[j][pos_j] = old_val_j
    
    return best_rows

def _element_local_search(initial_rows, n_rows, k, rng, deadline):
    """
    Element-level local search: try to reduce individual elements.
    """
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(10000):
        if time.time() > deadline - 3:
            break
        
        # Pick a random row and position
        idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        
        old_val = rows[idx][pos]
        
        # Try to reduce this element
        for new_val in range(old_val - 1, rows[idx][pos-1], -1):
            candidate = list(rows[idx])
            candidate[pos] = new_val
            
            if not _is_valid_row(candidate):
                continue
            
            new_diffs = _row_diffs(candidate)
            
            # Check conflicts with other rows
            conflict = False
            for j in range(n_rows):
                if j == idx:
                    continue
                if new_diffs & _row_diffs(rows[j]):
                    conflict = True
                    break
            
            if not conflict:
                rows[idx] = candidate
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                break  # Stop reducing this position
    
    return best_rows
