import time
import random

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _solve_row(row_idx, current_row, used_diffs, k, max_scope, deadline):
    """
    Recursively fill the current row (starting with 0) with k more elements.
    Returns True if a valid row is found, False otherwise.
    """
    if time.time() > deadline:
        return False
    
    if len(current_row) == k + 1:
        return True
    
    last = current_row[-1]
    # We need to choose the next element x > last
    # x must be <= max_scope
    # Diffs: x - y for all y in current_row must not be in used_diffs
    
    # Heuristic: try smaller x first to keep scope low
    # But we might need to jump.
    # Lower bound: last + 1
    # Upper bound: max_scope
    
    for x in range(last + 1, max_scope + 1):
        # Check diffs
        valid = True
        new_diffs = []
        for y in current_row:
            d = x - y
            if d in used_diffs:
                valid = False
                break
            # Since we are building the row in increasing order, 
            # and used_diffs contains all diffs from previous rows and 
            # already placed elements in this row? 
            # Wait, used_diffs is global. 
            # We also need to ensure new_diffs don't conflict with each other?
            # No, differences between elements in the SAME row are unique 
            # because the row is strictly increasing. 
            # However, we must ensure that the *new* differences don't 
            # collide with differences *already in this row*?
            # Actually, if we add x, the new diffs are x-y for y in row.
            # These are all distinct from each other because y are distinct.
            # They must not be in used_diffs (which includes previous rows 
            # and previous steps of this row if we updated it, but we don't 
            # update used_diffs until the row is complete in the outer loop? 
            # No, it's better to check against a local set + global set).
            
            # Let's refine: 
            # used_diffs contains ALL diffs from completed rows.
            # We need to check against used_diffs AND diffs generated 
            # by current_row elements so far?
            # If we just check `d in used_diffs`, and we haven't added 
            # current_row's internal diffs to used_diffs yet, we might 
            # miss a conflict within the same row?
            # No, a Golomb ruler requires all pairwise differences to be distinct.
            # If we pick x, d1 = x-y1, d2 = x-y2. d1 != d2 because y1 != y2.
            # Do d1 or d2 conflict with existing diffs in current_row?
            # Existing diffs in current_row are y_j - y_i.
            # If x - y_i == y_j - y_k, that's a conflict.
            # So we MUST check against the set of all diffs in the current row so far.
            
            pass
        
        # Correct logic:
        # Maintain `row_diffs` for the current row.
        # Check if any new diff is in `used_diffs` OR `row_diffs`.
        
        # Since this function is recursive, let's pass `row_diffs` or compute it.
        # For efficiency, compute new_diffs and check against used_diffs U row_diffs.
        
        row_diffs = set()
        n = len(current_row)
        for i in range(n):
            for j in range(i+1, n):
                row_diffs.add(current_row[j] - current_row[i])
        
        new_diffs = set()
        for y in current_row:
            d = x - y
            if d in used_diffs or d in row_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
            
        # Also check if new_diffs conflict with each other? 
        # No, as established, they are distinct.
        
        # Tentatively accept
        current_row.append(x)
        # We don't update used_diffs here yet because we might backtrack.
        # But we need to pass updated state to next recursion?
        # The function signature is fixed. 
        # Actually, _solve_row is called for one row. 
        # It returns True/False. 
        # If it returns True, the row is valid.
        
        if _solve_row(row_idx, current_row, used_diffs, k, max_scope, deadline):
            return True
            
        current_row.pop()
        
    return False

def _construct(n_rows, k, max_scope, deadline):
    """
    Try to construct n_rows rows within max_scope.
    """
    used_diffs = set()
    rows = []
    
    # Order of rows might matter. 
    # We will fill rows one by one.
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        current_row = [0]
        if not _solve_row(r, current_row, used_diffs, k, max_scope, deadline):
            return None
        
        # Extract diffs from this row to add to used_diffs
        for i in range(len(current_row)):
            for j in range(i+1, len(current_row)):
                used_diffs.add(current_row[j] - current_row[i])
        
        rows.append(current_row)
        
    return rows

def construct(n_rows, k):
    start_time = time.time()
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140
    
    # Lower bound for scope:
    # Total diffs needed: n_rows * (k*(k+1)/2)
    # Max diff in scope M is M.
    # So M >= n_rows * k*(k+1)/2.
    # For n=7, k=6: 7 * 21 = 147? 
    # Wait. k=6 means 7 points. Diffs per row = 7*6/2 = 21.
    # Total diffs = 7 * 21 = 147.
    # So M >= 147.
    # The prompt says "simple counting bound is 105". 
    # Let's re-read: "Choose 7 rows of 6 increasing integers starting at 0".
    # "6 increasing integers starting at 0" -> 0, a1, a2, a3, a4, a5. 
    # That is 6 integers. k=6 in the prompt description "k+1 integers".
    # If k=6, we have 7 integers? 
    # "rows of 6 increasing integers starting at 0" -> length 6.
    # "list of k + 1 integers" -> if length is 6, k=5?
    # Let's check the prompt: "Choose 7 rows of 6 increasing integers starting at 0".
    # "Each row is a list of k + 1 integers".
    # If the row has 6 integers, then k+1 = 6 => k=5.
    # Let's verify the counting bound 105.
    # If k=5 (6 points), diffs per row = 6*5/2 = 15.
    # Total diffs = 7 * 15 = 105.
    # So k=5.
    
    # The function signature is `construct(n_rows, k)`.
    # The evaluator passes `n_rows` and `k`.
    # Based on "7 rows of 6 increasing integers", the evaluator likely passes `k=5` 
    # if it follows "k+1 integers", OR `k=6` if it considers the count of integers as k.
    # Standard Golomb ruler notation: order n means n marks.
    # Here "k+1 integers". If the text says "6 integers", then k+1=6 => k=5.
    # However, often "k" in these problems refers to the number of marks or the order.
    # Let's look at the failed code: `while len(row) < k + 1`.
    # If the evaluator passes k=6, the code expects 7 integers.
    # If the evaluator passes k=5, the code expects 6 integers.
    # The prompt says "Choose 7 rows of 6 increasing integers".
    # It also says "The evaluator passes this task's n_rows and k".
    # It is highly probable that `k` corresponds to the parameter in the description 
    # such that `k+1` is the length. So if length is 6, k=5.
    # BUT, the failed code used `k+1`. If the evaluator passed k=6, it would look for 7.
    # If the evaluator passed k=5, it would look for 6.
    # The error "difference 8 occurs twice" implies it ran and produced output.
    # Let's assume the evaluator passes the `k` such that we generate `k+1` items.
    # If the task is "6 integers", k=5.
    
    # Let's calculate the lower bound dynamically.
    # Points per row = k + 1.
    # Diffs per row = (k+1)*k / 2.
    # Total diffs = n_rows * (k+1)*k / 2.
    # Lower bound M >= Total diffs.
    
    num_points = k + 1
    diffs_per_row = num_points * (num_points - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    min_scope = total_diffs
    
    # Start from a slightly higher bound to ensure we find a solution quickly,
    # then try to lower it.
    # Greedy usually works for small M.
    
    # Try increasing bounds starting from min_scope
    # But we want the MINIMUM scope.
    # So we should try bounds from min_scope upwards? 
    # No, we want the smallest valid scope.
    # So we try M = min_scope, min_scope + 1, ... until we find a solution.
    
    # However, backtracking is expensive.
    # Let's try a range.
    
    # Initial guess: The prompt says best known is 112.
    # If k=5 (6 points), bound is 105. 112 is close.
    # If k=6 (7 points), bound is 147.
    
    # Let's assume the evaluator passes k such that we need to find the min scope.
    
    best_rows = None
    best_scope = float('inf')
    
    # We will try bounds from min_scope upwards.
    # If we find a solution at M, we can try to verify if M-1 is possible?
    # No, just find the first valid M.
    
    # To be safe against time limits, we will try a few bounds.
    # If k=5, min_scope=105.
    # If k=6, min_scope=147.
    
    # Let's run a search for M in [min_scope, min_scope + 20]
    # If found, return it.
    
    # Actually, a smarter approach:
    # Use randomized restarts with a fixed high bound first to get *a* solution,
    # then try to lower the bound.
    # But the prompt asks to minimize the largest entry.
    
    # Strategy:
    # 1. Try M = min_scope.
    # 2. If fail, M = min_scope + 1, etc.
    # This is efficient if a solution exists near the bound.
    
    # However, _construct is deterministic (greedy-ish backtracking).
    # It might fail for M=min_scope even if a solution exists, due to bad ordering.
    # We can add randomization or multiple attempts.
    
    # Given 120s, we can afford some retries.
    
    # Let's try bounds from min_scope to min_scope + 100.
    # If we find a solution at M, we return it.
    # Since we iterate M upwards, the first solution found is the minimum M 
    # (for this specific solver's capability).
    
    # To improve success rate, we can shuffle the order of trying candidates 
    # or restart with different random seeds for the backtracking.
    
    # Let's modify _solve_row to accept a random seed or use a global RNG.
    
    rng = random.Random(42)
    
    # We will try to find a solution for increasing M.
    # For each M, we try a few times with different randomizations in the search.
    
    # Since _solve_row is currently deterministic (iterates x from low to high),
    # it might get stuck.
    # Let's add a randomization option to _solve_row?
    # Or just rely on the fact that for small k, the search space is small.
    
    # For k=5 (6 points), the search is very fast.
    # For k=6 (7 points), it might be slower.
    
    # Let's assume the standard case is small enough.
    
    for M in range(min_scope, min_scope + 50):
        if time.time() > deadline - 5:
            break
            
        # Try to construct with max_scope M
        # We try a few times with different "strategies" (e.g. shuffling candidates)
        # But _solve_row is currently pure.
        # Let's just try it once. If it fails, try M+1.
        
        sol = _construct(n_rows, k, M, deadline - 5)
        
        if sol is not None:
            # Verify
            if _validate(sol):
                return sol
            else:
                # Should not happen if _construct is correct
                print("Validation failed for constructed solution")
                
    # If we didn't find a solution in the range, try a higher bound with more time
    # Or return the best found (which is None).
    # Fallback: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    return rows
