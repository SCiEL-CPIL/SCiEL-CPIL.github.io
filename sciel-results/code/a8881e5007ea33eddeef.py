"""Difference triangle set (4,7) - randomized backtracking with scope constraints."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill
    
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope: need at least total_diffs_needed distinct differences
    # in range 1..scope, so scope >= total_diffs_needed
    lb = total_diffs_needed
    
    best_rows = None
    best_scope = None
    
    def validate(rows):
        """Check that all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def scope_of(rows):
        return max(max(r) for r in rows) if rows else 0
    
    def backtrack(rows, used_diffs, row_idx, col_idx, current_row, scope, deadline_t):
        """
        Backtrack to build rows one at a time.
        rows: completed rows so far
        used_diffs: set of all differences used
        row_idx: which row we're building
        col_idx: which column in current row
        current_row: the row being built (list)
        scope: maximum allowed value
        """
        if time.time() > deadline_t:
            return None
        
        if row_idx == n_rows:
            return rows + [current_row]
        
        if col_idx == k + 1:
            return backtrack(rows + [current_row], used_diffs, row_idx + 1, 0, [], scope, deadline_t)
        
        # Determine the range of candidate values for this position
        if col_idx == 0:
            lo, hi = 0, 0
        else:
            lo = current_row[-1] + 1
            hi = scope
        
        if lo > hi:
            return None
        
        # Generate candidates: for each candidate x, check if all differences x - a for a in current_row are unused
        # For efficiency, try candidates in a smart order
        candidates = []
        for x in range(lo, hi + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append((x, new_diffs))
        
        if not candidates:
            return None
        
        # For the first column, we must pick 0
        if col_idx == 0:
            # x must be 0
            if 0 in [c[0] for c in candidates]:
                idx = [c[0] for c in candidates].index(0)
                x, new_diffs = candidates[idx]
                result = backtrack(rows, used_diffs, row_idx, col_idx + 1, current_row + [x], scope, deadline_t)
                if result is not None:
                    return result
            return None
        
        # Randomize order of candidates for diversity
        random.shuffle(candidates)
        
        for x, new_diffs in candidates:
            result = backtrack(rows, used_diffs | new_diffs, row_idx, col_idx + 1, current_row + [x], scope, deadline_t)
            if result is not None:
                return result
        
        return None
    
    def backtrack_with_order(rows, used_diffs, row_idx, col_idx, current_row, scope, deadline_t, order_hint):
        """Backtrack with a preferred ordering for candidates."""
        if time.time() > deadline_t:
            return None
        
        if row_idx == n_rows:
            return rows + [current_row]
        
        if col_idx == k + 1:
            return backtrack_with_order(rows + [current_row], used_diffs, row_idx + 1, 0, [], scope, deadline_t, order_hint)
        
        if col_idx == 0:
            if 0 not in current_row:
                result = backtrack_with_order(rows, used_diffs, row_idx, col_idx + 1, current_row + [0], scope, deadline_t, order_hint)
                if result is not None:
                    return result
            return None
        
        lo = current_row[-1] + 1
        hi = scope
        
        if lo > hi:
            return None
        
        candidates = []
        for x in range(lo, hi + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append((x, new_diffs))
        
        if not candidates:
            return None
        
        # Sort by some heuristic: prefer smaller x (tighter packing) but with some randomness
        if order_hint == 'small':
            candidates.sort(key=lambda c: c[0])
            # Add slight randomization
            if len(candidates) > 1:
                swap_range = min(3, len(candidates))
                for i in range(swap_range):
                    j = random.randint(i, swap_range - 1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
        elif order_hint == 'large':
            candidates.sort(key=lambda c: -c[0])
            if len(candidates) > 1:
                swap_range = min(3, len(candidates))
                for i in range(swap_range):
                    j = random.randint(i, swap_range - 1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
        else:
            random.shuffle(candidates)
        
        for x, new_diffs in candidates:
            result = backtrack_with_order(rows, used_diffs | new_diffs, row_idx, col_idx + 1, current_row + [x], scope, deadline_t, order_hint)
            if result is not None:
                return result
        
        return None
    
    # Start with a greedy solution as initial best
    def greedy_solve(scope):
        """Try to build a solution with given scope using greedy + limited backtracking."""
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for c in range(1, k + 1):
                found = False
                for x in range(row[-1] + 1, scope + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    # First, get a baseline greedy solution
    greedy_scope = lb + 50  # generous initial bound
    greedy_result = greedy_solve(greedy_scope)
    if greedy_result and validate(greedy_result):
        best_rows = greedy_result
        best_scope = scope_of(greedy_result)
    
    # Now try to improve with backtracking, reducing scope
    # Try various scopes from low to high
    current_scope = lb
    max_scope_tried = best_scope if best_scope else lb + 200
    
    attempt = 0
    while time.time() < deadline and current_scope <= max_scope_tried:
        attempt += 1
        
        # Try a few random restarts at this scope
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
            
            order_hint = random.choice(['small', 'large', 'random'])
            result = backtrack_with_order([], set(), 0, 0, [], current_scope, deadline, order_hint)
            
            if result is not None:
                if validate(result):
                    s = scope_of(result)
                    if best_rows is None or s < best_scope:
                        best_rows = result
                        best_scope = s
                    found = True
                    break
        
        if found:
            # Found a solution at this scope, try to go lower
            current_scope = lb
            max_scope_tried = best_scope - 1
        else:
            current_scope += 1
    
    # If we still have time, try to improve further with more aggressive search
    if best_rows is not None and time.time() < deadline - 5:
        # Try to find a solution with scope < best_scope
        for target_scope in range(lb, best_scope):
            if time.time() > deadline:
                break
            for restart in range(5):
                if time.time() > deadline:
                    break
                order_hint = random.choice(['small', 'large', 'random'])
                result = backtrack_with_order([], set(), 0, 0, [], target_scope, deadline, order_hint)
                if result is not None and validate(result):
                    best_rows = result
                    best_scope = target_scope
                    break
            if best_scope == target_scope:
                break
    
    # Final fallback: if no valid solution found, use greedy
    if best_rows is None or not validate(best_rows):
        # Generate a valid solution with a larger scope
        scope = lb + 100
        while scope < lb + 500:
            if time.time() > deadline:
                break
            result = greedy_solve(scope)
            if result and validate(result):
                best_rows = result
                best_scope = scope
                break
            scope += 10
    
    if best_rows is None:
        # Last resort: very large scope greedy
        scope = lb + 200
        result = greedy_solve(scope)
        if result:
            best_rows = result
    
    return best_rows if best_rows else []
