"""Difference triangle set (7,5) - backtracking with heuristics and time guard."""
import time
import random
import sys

_DEADLINE = 145.0  # safety margin before 150s kill

def _validate(rows):
    """Check that all positive differences are distinct across all rows."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                if d in diffs:
                    return False
                diffs.add(d)
        used |= diffs
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k):
    """Original greedy: each row takes smallest next mark with unused diffs."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, time_limit, rng=None):
    """
    Backtracking search to minimize the maximum entry.
    We build rows one at a time, each row is a sequence of k+1 integers starting at 0.
    We use a greedy-with-backtracking per row, and backtrack across rows.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_rows = None
    best_scope = float('inf')
    
    used_diffs = set()
    rows = []
    
    # We'll try to build rows in a specific order.
    # For each row, we try candidate starting sequences.
    
    def _time_exceeded():
        return time.time() > deadline
    
    def _try_place_row(used_diffs, rows, row_idx, max_val_hint):
        """
        Try to place row row_idx. Returns (success, new_used_diffs, new_rows, scope_of_this_row).
        Uses backtracking within the row.
        """
        if _time_exceeded():
            return None
        
        row = [0]
        used_in_row = set()
        
        def _place_element(row, used_in_row, pos):
            """Place element at position pos in the row."""
            if pos == k + 1:
                # Row complete
                return (True, row[:], used_in_row.copy())
            
            if _time_exceeded():
                return None
            
            # Try candidate values for row[pos]
            # Start from row[-1] + 1
            start_val = row[-1] + 1
            # Upper bound: we want to keep scope small
            # max_val_hint is our current best scope; we shouldn't exceed it
            end_val = min(max_val_hint, start_val + 200)  # reasonable window
            
            # Collect candidates and shuffle for diversity
            candidates = list(range(start_val, end_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check if adding val creates any duplicate diff
                new_diffs = set()
                valid = True
                for prev in row:
                    d = val - prev
                    if d in used_diffs or d in used_in_row:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(val)
                used_in_row.update(new_diffs)
                
                result = _place_element(row, used_in_row, pos + 1)
                row.pop()
                
                if result is not None:
                    return result
                
                used_in_row.difference_update(new_diffs)
            
            return None
        
        result = _place_element(row, used_in_row, 1)
        if result is None:
            return None
        else:
            _, complete_row, row_diffs = result
            new_used = used_diffs | row_diffs
            new_rows = rows + [complete_row]
            return (True, new_used, new_rows, max(complete_row))
    
    # Main backtracking: try to place all rows
    # We'll do a simple DFS with restarts
    
    attempts = 0
    while not _time_exceeded():
        attempts += 1
        used_diffs = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            if _time_exceeded():
                success = False
                break
            
            # For each row, try to find a valid placement
            # Use a greedy approach first, then backtrack if needed
            result = _try_place_row(used_diffs, rows, row_idx, best_scope)
            
            if result is None:
                # Backtrack: try a different approach
                # Simple strategy: restart with a fresh random seed
                success = False
                break
            
            _, new_used, new_rows, scope = result
            used_diffs = new_used
            rows = new_rows
            
            if scope >= best_scope and len(rows) == n_rows:
                # This row's scope already exceeds best, but we might still improve
                # Actually, if this intermediate scope is already >= best, we can prune
                # But we need all rows. Let's continue but note this.
                pass
        
        if success and len(rows) == n_rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows[:]
                # If we hit the theoretical lower bound, we can stop
                # Lower bound: n_rows * (k+1) * k / 2 distinct diffs, max diff = m
                # Number of distinct diffs available in [1, m] is m*(m+1)/2
                # We need n_rows * (k+1) * k / 2 distinct positive diffs
                # So m*(m+1)/2 >= n_rows * (k+1) * k / 2
                # m >= sqrt(n_rows * (k+1) * k) roughly
                # For n_rows=7, k=5: 7*6*5/2 = 105 diffs needed, so m*(m+1)/2 >= 105
                # m=14: 14*15/2=105, so m >= 14 is the counting bound per row? No...
                # Actually the counting bound is on the total number of distinct differences.
                # Total distinct differences needed: n_rows * C(k+1, 2) = 7 * 15 = 105
                # Max difference is m, so we need at least 105 distinct values in [1, m]
                # So m >= 105 is a trivial bound? No, that's not right either.
                # The differences can repeat across different pairs as long as they're in different rows? 
                # No, the constraint is all differences are distinct over ALL rows.
                # So we need 105 distinct positive integers, each <= m.
                # So m >= 105. But the best known is 112, and counting bound is 105.
                # So if we reach 105, we're optimal.
                if scope <= 105:
                    break
        
        # If we found a good solution, we might try to improve further
        # but with limited time, let's do a few more restarts
    
    if best_rows is not None:
        return best_rows
    else:
        # Fallback to greedy
        return _greedy_construct(n_rows, k)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all differences distinct across all rows.
    """
    start_time = time.time()
    
    # Quick greedy baseline
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    # Time budget for backtracking search
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        return greedy_rows
    
    # Run backtracking search with time limit
    # Use multiple random seeds for diversity
    best_result = greedy_rows
    best_scope = greedy_scope
    
    search_time = min(remaining - 2, 100)  # leave 2s margin
    
    # Try a few different random seeds
    for seed in [42, 123, 7, 999, 2025]:
        if time.time() > _DEADLINE - 2:
            break
        rng = random.Random(seed)
        result = _backtracking_search(n_rows, k, search_time / 3, rng)
        if result is not None:
            scope = _scope(result)
            if scope < best_scope:
                best_scope = scope
                best_result = result
    
    return best_result
