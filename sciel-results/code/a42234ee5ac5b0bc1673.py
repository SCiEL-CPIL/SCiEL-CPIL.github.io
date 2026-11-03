import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _csp_backtrack_row(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
    """
    Backtrack to build a single valid row of length k+1 starting at 0,
    avoiding conflicts with used_diffs and internal conflicts.
    Uses forward checking: at each step, verify that remaining positions
    can still be filled.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    min_next = prev + 1
    # We need to fit (n - pos - 1) more elements after this one.
    # So the current element must be at most max_val - (n - pos - 1)
    max_next = max_val - (n - pos - 1)
    
    if min_next > max_next:
        return None
    
    # Generate candidates
    # We want to try smaller values first to keep scope low, 
    # but randomize slightly to avoid getting stuck in local patterns.
    candidates = list(range(min_next, max_next + 1))
    
    # Heuristic: Try values that create "sparse" differences first?
    # For now, just shuffle the upper half to allow some exploration.
    if len(candidates) > 10:
        mid = len(candidates) // 2
        rng.shuffle(candidates[mid:])
        
    for val in candidates:
        if time.time() > deadline:
            return None
        
        # Check conflicts with current row's diffs and used_diffs
        conflict = False
        new_diffs = []
        for idx in range(pos):
            d = val - current_row[idx]
            if d in current_diffs or d in used_diffs:
                conflict = True
                break
            new_diffs.append(d)
        
        if conflict:
            continue
        
        current_row.append(val)
        current_diffs.update(new_diffs)
        
        result = _csp_backtrack_row(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)
        
        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)
        
        if result is not None:
            return result
    
    return None

def _construct_rows_csp(n_rows, k, deadline, rng, target_scope):
    """
    Construct n_rows valid rows using CSP backtracking with forward checking.
    This is the core structural change: instead of SA repair, we do exact
    search for valid rows given the global constraint of used differences.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        best_row = None
        best_max_val = float('inf')
        
        # Try multiple attempts with different random orderings if the first fails
        # or if we want to find a tighter row.
        # Since we are minimizing scope, we try to find the row with the smallest possible max value.
        # However, greedy smallest-first usually works well for CSP if feasible.
        
        # We use a binary search on the max value of the row?
        # Or just try to build it with the current target_scope.
        # If target_scope is too tight, it fails.
        
        # Let's try to build a row that fits within target_scope.
        # To allow for better packing, we might want to try slightly looser bounds first?
        # No, we want the global scope to be small.
        
        # Simple strategy: Try to build a row with max_val = target_scope.
        # If it fails, we might need to backtrack to previous rows.
        # For simplicity in this phase, we just try to build it.
        # If it fails, we return None and the caller might try a larger target_scope.
        
        current_row = [0]
        current_diffs = set()
        
        result = _csp_backtrack_row(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)
        
        if result is None:
            return None
        
        rows.append(result)
        for d in _diffs_for_row(result):
            used_diffs.add(d)
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: CSP Backtracking Construction
    # We know the best known scope is 112. The counting bound is 105.
    # We try scopes starting from a tight bound and increasing.
    # We also try a few random seeds/restarts if the first one fails.
    
    # Estimate a lower bound.
    # Each row has k*(k+1)/2 differences.
    # Total differences = n_rows * k*(k+1)/2.
    # These must be distinct values in [1, scope].
    # So scope >= n_rows * k*(k+1)/2.
    lb = n_rows * k * (k + 1) // 2
    
    # Known best is 112 for (7,5).
    # Let's try a range of target scopes.
    # We start from the lower bound and go up.
    
    # To save time, we can jump to known good ranges if we have them, 
    # but since this is a general interface, we use the bound.
    
    # If lb is very low (e.g. 105), we try 105, 106, ...
    # If lb is high, we start there.
    
    # Optimization: If the CSP search for a specific scope fails quickly,
    # we increment the scope.
    
    # We will try scopes from max(lb, 100) to 150.
    start_scope = max(lb, 100)
    end_scope = 150
    
    # If we have a heuristic upper bound (e.g. from greedy), use it to cap.
    greedy_rows = _fast_greedy(n_rows, k, rng)
    greedy_scope = _scope(greedy_rows) if greedy_rows else end_scope
    
    # We want to find the minimum scope.
    # So we iterate from start_scope to min(end_scope, greedy_scope).
    
    # To ensure we don't spend too much time on a single scope,
    # we allocate time per scope attempt.
    num_scopes = min(end_scope, greedy_scope) - start_scope + 1
    if num_scopes <= 0:
        num_scopes = 1
        start_scope = max(lb, 100)
    
    time_per_scope = (deadline - time.time() - 5.0) / max(1, num_scopes)
    
    found_valid = False
    for target_scope in range(start_scope, min(end_scope, greedy_scope) + 1):
        if time.time() > deadline - 5.0:
            break
            
        scope_deadline = time.time() + time_per_scope
        
        # Try a few random restarts for this scope
        # CSP is deterministic given the order, but we randomize candidate order.
        # So different seeds might find different solutions.
        
        rows = _construct_rows_csp(n_rows, k, scope_deadline, rng, target_scope)
        
        if rows is not None:
            s = _scope(rows)
            # It's possible the row's max value is less than target_scope
            # because we just capped the search space.
            if s < best_score:
                best_score = s
                best_result = rows
            found_valid = True
            # Since we are increasing scope, and we found a valid one,
            # this is the minimal scope for this specific search path.
            # However, a different random seed might find a solution with a smaller scope?
            # No, we are iterating scope upwards. The first valid solution found
            # for a given scope has scope <= target_scope.
            # If we found one at target_scope, can we find one at target_scope - 1?
            # Yes, if the previous iteration failed due to bad luck (randomization).
            # So we should continue to check smaller scopes? 
            # No, we are iterating upwards. 
            # Wait, if we found a valid solution with scope 112, we should check if 111 is possible.
            # But we already checked 111 and it failed.
            # The failure might be due to the specific random choices.
            # So we should probably do more restarts for the *current* best found scope?
            # Or just accept the first valid one found as a good upper bound.
            
            # Given the time constraint, accepting the first valid solution found
            # while iterating from low to high is a solid strategy.
            # We break because any higher scope is worse.
            # And lower scopes already failed (with some probability of failure).
            # To be safe, we can do a few more restarts at this scope to see if we can
            # find a "tighter" row (though scope is capped by target_scope anyway).
            # Actually, if scope is 112, it might be 110. 
            # But the search was capped at 112.
            # So the result is valid and has scope <= 112.
            # We break to save time for potential local optimization?
            # The prompt says "largest entry as small as possible".
            # CSP finds an exact solution for the constraint "all diffs distinct".
            # So the scope is exactly the max entry of the found rows.
            
            break
        
    # If CSP failed for all scopes in range, use greedy
    if best_result is None:
        if greedy_rows is not None:
            best_result = greedy_rows
            best_score = _scope(greedy_rows)
        else:
            # Last resort: simple sequential (should not happen if greedy works)
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                rows.append(row)
            best_result = rows
            best_score = _scope(rows)
            
    # Phase 2: Local Optimization (Optional)
    # If we have time, try to reduce the scope of the found solution.
    # Since CSP found a valid solution, we can try to "shrink" it.
    # But CSP already minimized the scope in the search space.
    # However, the search space was capped at `target_scope`.
    # If `target_scope` was 112, the solution might have max 110.
    # Can we make it 109?
    # We would need to re-run CSP with target_scope=109.
    # But we already tried 109 and it failed.
    # So likely 110 is the minimum for this set of rows.
    
    # We can try to swap rows or re-order them? No, rows are independent.
    
    # We can try to change the specific values to reduce the max.
    # This is effectively re-running CSP with a tighter bound.
    # Let's do a few quick restarts with the current best_score - 1, - 2, etc.
    
    if time.time() < deadline - 10:
        # Try to improve by 1 or 2
        for delta in [1, 2, 3]:
            new_target = best_score - delta
            if new_target < start_scope:
                break
            if time.time() > deadline - 5:
                break
            
            # Short time budget for improvement attempts
            imp_deadline = time.time() + 2.0
            # Use a different seed for variation
            imp_rng = random.Random(12345 + delta)
            
            rows = _construct_rows_csp(n_rows, k, imp_deadline, imp_rng, new_target)
            if rows is not None:
                s = _scope(rows)
                if s < best_score:
                    best_score = s
                    best_result = rows
                # If we found one, we can stop trying smaller deltas for this pass
                # or continue to see if even smaller is possible.
                # Given time, break.
                break
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
    
    return best_result
