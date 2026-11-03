import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Lower bound: n_rows * k * (k+1) / 2
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Get a baseline solution using a greedy approach
    baseline_solution, baseline_scope = greedy_solve(n_rows, k)
    
    if not baseline_solution:
        return []
        
    best_solution = baseline_solution
    best_scope = baseline_scope
    
    # If we hit the lower bound, we can't do better
    if best_scope <= lower_bound:
        return best_solution

    # Search for better solutions starting from the lower bound
    # We try M from lower_bound up to best_scope - 1
    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break
            
        # Try to find a solution with max entry <= M
        # We use a randomized backtracking search with symmetry breaking
        # and bitset optimization.
        
        # Adaptive time limit for each M
        # If we are close to the lower bound, give it more time.
        # If we are close to the current best, give it less time.
        time_budget = 10.0
        # Simple heuristic: give more time to smaller M
        if M < lower_bound + 5:
            time_budget = 20.0
        elif M < lower_bound + 20:
            time_budget = 10.0
        else:
            time_budget = 5.0
            
        sol = solve_dts_optimized(n_rows, k, M, deadline, time_budget)
        
        if sol is not None:
            best_solution = sol
            best_scope = M
            # Since we are searching in increasing M, the first solution found
            # for a given M is the minimum possible for this iteration.
            # We can return immediately because we want the smallest M.
            return sol

    return best_solution

def greedy_solve(n_rows, k):
    """
    Fast greedy construction to get a valid baseline solution.
    """
    used_diffs = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                conflict_mask = 0
                for a in row:
                    diff = x - a
                    if diff > 0:
                        conflict_mask |= (1 << diff)
                
                if (used_diffs & conflict_mask) == 0:
                    row.append(x)
                    used_diffs |= conflict_mask
                    break
                x += 1
        rows.append(row)
        
    return rows, max(max(row) for row in rows)

def solve_dts_optimized(n_rows, k, max_scope, deadline, time_budget):
    """
    Optimized backtracking with symmetry breaking and bitsets.
    """
    rng = random.Random()
    
    # To handle symmetry breaking, we enforce that rows are in lexicographic order.
    # Row i must be <= Row i+1 lexicographically.
    
    def backtrack(row_idx, pos, current_last, row_so_far, row_mask, prev_row):
        # Time check
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            # Check symmetry breaking against previous row
            if row_idx > 0:
                if not is_lex_leq(row_so_far, prev_row):
                    return None
            return row_so_far
            
        # Pruning: Can we fit the remaining numbers?
        # We need (k+1 - pos) more numbers.
        # The smallest they can be is current_last + 1, ..., current_last + (k+1-pos)
        # The largest of these is current_last + (k+1-pos)
        # This must be <= max_scope
        if current_last + (k + 1 - pos) > max_scope:
            return None
            
        # Max value for the next number x
        # We need (k - pos) more numbers after x.
        # So x + (k - pos) <= max_scope
        max_x = max_scope - (k - pos)
        
        # Candidates
        candidates = []
        for x in range(current_last + 1, max_x + 1):
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            if ((used_diffs | row_mask) & conflict_mask) == 0:
                candidates.append((x, conflict_mask))
                
        if not candidates:
            return None
            
        # Randomize order to escape local minima
        rng.shuffle(candidates)
        
        for x, cm in candidates:
            new_row_mask = row_mask | cm
            new_row = row_so_far + [x]
            
            res = backtrack(row_idx, pos + 1, x, new_row, new_row_mask, prev_row)
            if res is not None:
                return res
                
        return None

    used_diffs = 0
    rows = []
    
    # Try multiple times with different random seeds/initializations if needed
    # But for now, one run with randomization inside.
    # To increase robustness, we can try a few times with different shuffles.
    
    # We build rows one by one.
    # Symmetry breaking: Row 0 is free. Row 1 must be >= Row 0. etc.
    
    # To speed up, we can try to fill the rows in a way that balances the load.
    # But sequential is fine for now.
    
    # Let's add a restart mechanism inside the function if it fails quickly?
    # No, the caller will call it again for the same M if it fails?
    # The caller loops M. If it fails for M, it moves to M+1.
    # We should try multiple times for the SAME M before giving up.
    
    attempts = 0
    max_attempts = 5 # Adjust based on time budget
    
    while attempts < max_attempts and time.time() < deadline:
        # Reset state for new attempt
        used_diffs = 0
        rows = []
        
        # Re-define backtrack closure to use current used_diffs?
        # No, used_diffs is modified in place. We need to reset it.
        # The closure captures `used_diffs` by reference.
        # So we must reset `used_diffs = 0` before each attempt.
        
        # Wait, the `backtrack` function above uses `used_diffs` from the outer scope.
        # If we call it multiple times, we need to ensure `used_diffs` is reset.
        # The current structure defines `backtrack` inside `solve_dts_optimized`.
        # So we can just reset `used_diffs` and `rows` in the loop.
        
        # However, `backtrack` also needs to know `prev_row` for symmetry breaking.
        # The first row has no previous row.
        
        # Let's restructure slightly to allow multiple attempts.
        
        found = False
        for i in range(n_rows):
            # prev_row is rows[-1] if i>0 else None
            prev_row = rows[-1] if i > 0 else None
            
            # We need to pass prev_row to backtrack.
            # The current `backtrack` signature takes `prev_row`.
            
            row = backtrack(i, 1, 0, [0], 0, prev_row)
            if row is None:
                found = False
                break
            rows.append(row)
            
            # Update global used_diffs
            for j in range(len(row)):
                for l in range(j+1, len(row)):
                    d = row[l] - row[j]
                    used_diffs |= (1 << d)
                    
            if time.time() > deadline:
                found = False
                break
                
        if found:
            return rows
            
        attempts += 1
        
    return None

def is_lex_leq(a, b):
    """
    Returns True if list a is lexicographically less than or equal to list b.
    """
    for i in range(len(a)):
        if a[i] < b[i]:
            return True
        if a[i] > b[i]:
            return False
    return True # Equal
