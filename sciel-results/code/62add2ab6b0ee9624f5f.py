import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 148  # Safety margin for 150s limit

    # Total number of distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound for scope M
    # We need total_diffs_needed distinct positive integers <= M
    # So M >= total_diffs_needed
    M_lb = total_diffs_needed

    # Upper bound heuristic: Greedy construction gives a valid upper bound
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    if best_scope == M_lb:
        return best_rows

    # Backtracking Search with Forward Checking
    # State:
    # rows: list of lists, partially filled
    # used_diffs: set of all differences used so far
    
    # We search for a solution with scope M.
    # We try increasing M starting from M_lb, but since M_lb is often too tight,
    # we start from a reasonable lower bound and go up, or just try to find ANY
    # valid solution with scope < best_scope.
    
    # Strategy:
    # 1. Determine a target scope M_target. We can start with best_scope - 1 and decrease?
    #    No, backtracking is expensive.
    #    Better: Binary search or linear search on M?
    #    Given the small search space (4 rows, 8 elements), we can try to find
    #    the minimal M directly using a solver that minimizes the max value.
    #    
    #    However, standard backtracking doesn't minimize naturally.
    #    We can use "Branch and Bound" or simply try M values.
    #    
    #    Let's try M values from M_lb to best_scope - 1.
    #    For each M, run a backtracking solver.
    #    If a solution is found, update best_scope and continue to lower M?
    #    Actually, if we find a solution for M, we can stop and return it?
    #    No, we want the *smallest* M.
    #    So we should search for the smallest M.
    #    
    #    But searching for M=112 might fail quickly or take long.
    #    Searching for M=133 (best known) is likely to succeed fast.
    #    
    #    Let's try to find a solution with scope < current best_scope.
    #    We can use a randomized restart strategy with different M values.
    #    
    #    Alternative: Use the backtracking solver to find *any* valid solution
    #    with a constraint that the last element of the last row is <= M.
    #    We can set M = best_scope - 1 initially. If found, set M = M - 1 and repeat.
    #    This is "iterative deepening" on the scope.

    def solve_for_scope(M, time_limit, seed):
        random.seed(seed)
        
        # State representation
        # rows[r] is a list of length k+1.
        # We fill rows row by row? Or element by element?
        # Filling row by row is simpler for forward checking within the row.
        # But global constraints (used_diffs) couple all rows.
        
        # Let's fill element by element, but prioritize completing a row.
        # Or better: Fill rows one by one.
        # Row 0 is fixed to start at 0.
        
        # Heuristic: MRV (Minimum Remaining Values) for the next element.
        # For a partially filled row, the next element x must be > prev.
        # The set of valid x is {prev+1, ..., M} minus those that create conflicts.
        
        # To make this fast, we need to efficiently check conflicts.
        # used_diffs is a set.
        
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        # Initialize first element of each row to 0? No, we fill sequentially.
        # Let's say we are filling row r, index c.
        # rows[r][0] is always 0.
        
        used_diffs = set()
        
        # Pre-compute potential conflicts? Hard.
        
        # Backtracking function
        # r: current row index
        # c: current column index
        # If c == k+1, move to next row.
        # If r == n_rows, solution found.
        
        # Optimization: Symmetry breaking?
        # Rows are distinguishable only by their content, but the set of rows is unordered?
        # No, the problem says "Choose 4 rows". The order of rows doesn't matter for validity.
        # So we can enforce an ordering on the rows to break symmetry.
        # E.g., rows[0] <= rows[1] <= ... in lexicographic order?
        # Or just sort the final rows.
        # Enforcing lexicographic order during search:
        # rows[r] must be lexicographically >= rows[r-1].
        # This reduces search space by n_rows! factor.
        
        def is_lex_greater_or_equal(row_a, row_b):
            # row_a >= row_b lexicographically
            for i in range(len(row_a)):
                if row_a[i] > row_b[i]:
                    return True
                if row_a[i] < row_b[i]:
                    return False
            return True # Equal
        
        def backtrack(r, c):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                # Move to next row
                return backtrack(r + 1, 0)
            
            # If c == 0, value is 0.
            if c == 0:
                rows[r][0] = 0
                # Check symmetry: if r > 0, rows[r] must be >= rows[r-1]
                # Since rows[r][0]=0 and rows[r-1][0]=0, we need rows[r] >= rows[r-1]
                # This is checked when we fill subsequent elements.
                # Actually, we should check the constraint as we fill.
                # For now, just recurse.
                return backtrack(r, c + 1)
            
            # Determine range for rows[r][c]
            prev_val = rows[r][c - 1]
            # Next value must be > prev_val
            # Also, if r > 0, rows[r][c] must be >= rows[r-1][c] (to maintain lex order)
            # Wait, lex order is on the whole row.
            # If rows[r][c-1] > rows[r-1][c-1], then rows[r] is already > rows[r-1] regardless of rest.
            # If rows[r][c-1] == rows[r-1][c-1], then rows[r][c] must be >= rows[r-1][c].
            
            min_val = prev_val + 1
            if r > 0:
                if rows[r][c-1] == rows[r-1][c-1]:
                    min_val = max(min_val, rows[r-1][c])
                # If rows[r][c-1] < rows[r-1][c-1], this branch is invalid (should have been pruned earlier)
                # If rows[r][c-1] > rows[r-1][c-1], min_val stays prev_val+1.
            
            # Max value is M
            max_val = M
            
            # Collect candidates
            # To use MRV, we'd want to know the number of valid values for each.
            # But here, the domain is contiguous.
            # We can just iterate.
            # Order of iteration: Smallest first? Largest first?
            # Smallest first tends to keep scope small, but might hit conflicts later.
            # Randomized order with restarts might be better.
            
            # Let's try iterating from min_val to max_val.
            # Optimization: Skip values that create immediate conflicts.
            
            for x in range(min_val, max_val + 1):
                # Check conflicts with previous elements in this row
                valid = True
                for i in range(c):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check symmetry constraint more rigorously?
                # The min_val logic above handles the prefix constraint.
                # But what if rows[r][c-1] < rows[r-1][c-1]?
                # That would mean rows[r] < rows[r-1] lexicographically, which is invalid.
                # We should prune this earlier.
                if r > 0 and rows[r][c-1] < rows[r-1][c-1]:
                    return False # Prune entire branch
                
                # If we get here, x is a valid candidate
                rows[r][c] = x
                
                # Update used_diffs
                new_diffs = []
                for i in range(c):
                    d = x - rows[r][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                if backtrack(r, c + 1):
                    return True
                
                # Undo
                for d in new_diffs:
                    used_diffs.remove(d)
                rows[r][c] = 0
                
            return False
        
        return backtrack(0, 0)

    # Main loop: Try to find a solution with scope < best_scope
    # We start from M_lb and go up? Or from best_scope - 1 and go down?
    # Going down is better if we expect to find a solution close to best_scope.
    # But if M_lb is 112 and best_scope is 134, searching 112..133 is 22 steps.
    # Each step is a backtracking search.
    # If the search for M=112 is very fast (fails quickly), this is good.
    # If it's slow, we waste time.
    
    # Let's try a hybrid:
    # 1. Try M_lb.
    # 2. If fail, try M_lb+1, etc.
    # 3. If time runs out, return best known.
    
    # However, backtracking can be very slow if M is too small.
    # We can use a time limit per M.
    
    current_M = M_lb
    seed_counter = 0
    
    # We also keep the best_rows from greedy as a fallback
    final_best_rows = best_rows
    final_best_scope = best_scope
    
    while current_M < final_best_scope:
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this M
        # If current_M is close to final_best_scope, we can spend more time?
        # Or just a fixed small time?
        # Let's give 5 seconds per M, but reduce if we are running out of time.
        time_for_this_M = min(5.0, deadline - time.time() - 2)
        if time_for_this_M < 0.1:
            break
        
        # Try to solve for this M
        # Use multiple seeds for randomized restarts?
        # The current backtrack is deterministic.
        # We can randomize the order of candidates.
        
        # Let's modify backtrack to accept a seed and randomize candidate order.
        # Actually, let's just run the deterministic one first.
        # If it fails, try randomized?
        
        # For now, let's just run the deterministic one.
        # To add randomness, we can shuffle the rows? No, symmetry breaking fixes order.
        # We can randomize the choice of x within the valid range?
        
        # Let's create a randomized version of backtrack.
        
        def backtrack_random(r, c, rng):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                return backtrack_random(r + 1, 0, rng)
            
            if c == 0:
                rows[r][0] = 0
                return backtrack_random(r, c + 1, rng)
            
            prev_val = rows[r][c - 1]
            min_val = prev_val + 1
            if r > 0:
                if rows[r][c-1] == rows[r-1][c-1]:
                    min_val = max(min_val, rows[r-1][c])
                if rows[r][c-1] < rows[r-1][c-1]:
                    return False
            
            max_val = M
            
            candidates = []
            for x in range(min_val, max_val + 1):
                valid = True
                for i in range(c):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                return False
            
            # Shuffle candidates
            rng.shuffle(candidates)
            
            for x in candidates:
                rows[r][c] = x
                
                new_diffs = []
                for i in range(c):
                    d = x - rows[r][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                if backtrack_random(r, c + 1, rng):
                    return True
                
                for d in new_diffs:
                    used_diffs.remove(d)
                rows[r][c] = 0
                
            return False
        
        # Try deterministic first
        found = solve_for_scope(current_M, deadline - 2, 0)
        
        if found:
            # Extract solution
            new_rows = [row[:] for row in rows]
            # Verify
            used = set()
            valid = True
            for r in new_rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in used:
                            valid = False
                        used.add(d)
            if valid and max(max(r) for r in new_rows) <= current_M:
                final_best_rows = new_rows
                final_best_scope = current_M
                # Continue to lower M?
                # We found a solution for M. We want the smallest M.
                # So we should continue searching for M-1?
                # But we are iterating M upwards from M_lb.
                # So if we found a solution for M, we can stop?
                # No, we want the *smallest* M.
                # If we are iterating M_lb, M_lb+1, ..., and we find a solution for M_lb+5,
                # we don't know if M_lb+1 was solvable but we just failed to find it quickly.
                # So we should continue?
                # But if M_lb+1 was infeasible, the search would have failed quickly.
                # If it was feasible but hard, we might have spent too much time.
                # Given the small range (112 to 134), we can afford to check each M.
                # So we just update best and continue to next M?
                # No, if we found a solution for M, we can set final_best_scope = M.
                # But we should try to find a smaller one.
                # Since we are going up, we haven't checked smaller ones yet?
                # Wait, we start at M_lb.
                # If M_lb is infeasible, we go to M_lb+1.
                # If M_lb+1 is feasible, we find it.
                # Then we stop? No, we want the *smallest*.
                # So the first M we find a solution for is the smallest?
                # Yes, because we are iterating in increasing order.
                # So we can break.
                
                # However, what if M_lb was feasible but we timed out?
                # Then we move to M_lb+1.
                # If M_lb+1 is feasible, we find it.
                # But M_lb might have been feasible.
                # So we should not break immediately?
                # But we have a time limit.
                # If we find a solution for M, we can return it.
                # It might not be optimal, but it's better than the greedy one.
                # To find the optimal, we need to be sure smaller M are infeasible.
                # Given the time constraint, finding *any* solution better than greedy is a win.
                # Let's break and return the first solution found in the upward sweep.
                
                break
        
        # If not found, try randomized?
        # If deterministic failed, randomized might find it if the search space is tricky.
        # But deterministic with symmetry breaking is quite strong.
        # Let's try a few randomized runs if time permits.
        
        if time.time() < deadline - 10:
            for seed in range(3):
                rng = random.Random(seed + current_M)
                # Reset state for randomized run
                rows = [[0] * (k + 1) for _ in range(n_rows)]
                used_diffs = set()
                
                found = backtrack_random(0, 0, rng)
                if found:
                    new_rows = [row[:] for row in rows]
                    used = set()
                    valid = True
                    for r in new_rows:
                        for i in range(len(r)):
                            for j in range(i+1, len(r)):
                                d = r[j] - r[i]
                                if d in used:
                                    valid = False
                                used.add(d)
                    if valid and max(max(r) for r in new_rows) <= current_M:
                        final_best_rows = new_rows
                        final_best_scope = current_M
                        break
            else:
                # If no break, continue to next M
                pass
            if final_best_scope == current_M:
                break
        
        current_M += 1
        
    return final_best_rows

# Test/Validation
if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
    # Validate
    used = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    valid = False
                used.add(d)
    print("Valid:", valid)
