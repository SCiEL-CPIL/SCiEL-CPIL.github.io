import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Problem parameters
    # Each row has k+1 elements: a_0=0, a_1, ..., a_k
    # Total differences per row: D = k*(k+1)//2
    # Total differences in system: T = n_rows * D
    # Differences must be distinct and in [1, M]
    # Lower bound for M: T (since we need T distinct integers from 1..M)
    # Upper bound estimate: Start with a greedy solution, then tighten.
    
    D = k * (k + 1) // 2
    T = n_rows * D
    
    # 1. Initial Greedy Solution to establish an upper bound (M_max)
    # We want to minimize M.
    def greedy_solution():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                # Try smallest possible value
                val = row[-1] + 1
                while True:
                    # Check conflicts with existing elements in this row
                    conflict = False
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        break
                    val += 1
                row.append(val)
                for i in range(len(row) - 1):
                    used_diffs.add(val - row[i])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    current_best_max = max(row[-1] for row in best_rows)
    
    # 2. Backtracking Solver with Forward Checking
    # We search for a configuration with max entry <= M.
    # Variables: The elements of the rows.
    # To make it efficient, we process rows one by one, and elements within a row sequentially.
    # However, a pure row-by-row backtracking is similar to the current program but with global backtracking.
    # A better CSP model: 
    # Variables: x_{r, j} for r in 0..n_rows-1, j in 1..k.
    # Domains: 
    #   x_{r, 1} in [1, M]
    #   x_{r, j} in [x_{r, j-1}+1, M]
    # Constraints:
    #   1. Strictly increasing in each row.
    #   2. All differences x_{r, j} - x_{r, i} (i < j) are distinct globally.
    
    # We will use a recursive backtracking algorithm that fills the grid.
    # To improve performance, we use a specific variable ordering.
    # Ordering: Fill rows in order? Or fill columns?
    # Filling rows in order is natural.
    # Within a row, fill elements in order.
    
    # State:
    # - current_row_idx
    # - current_col_idx (0 is fixed to 0)
    # - current_row_elements: list of elements in current row so far
    # - all_rows: list of completed rows
    # - used_diffs: set of all differences used so far
    
    # Optimization: 
    # If we are filling row r, element j, we can prune if the minimum possible max exceeds M.
    # Also, we can use "Forward Checking" to check if the next element has any valid moves?
    # For simplicity and robustness within time limits, we will use a randomized restart strategy
    # with a tight time limit per attempt, trying to beat the current best.
    
    def solve_with_limit(M_limit, time_limit):
        """
        Attempts to find a valid configuration with max entry <= M_limit.
        Returns the list of rows if found, else None.
        """
        # To add randomness and escape local minima (if we were doing local search),
        # but here it's exact search. Randomness helps if we randomize the order of candidates
        # or the order of rows.
        
        # We'll randomize the order of rows to be solved? 
        # No, rows are symmetric, so order doesn't matter for validity, but might for search speed?
        # Actually, since constraints are global (diffs), the order of processing rows matters for pruning.
        # Let's stick to row 0, 1, ... n_rows-1.
        
        # We need to be very fast.
        # Python recursion might be slow. We'll implement carefully.
        
        used_diffs = set()
        rows = []
        
        # Pre-allocate row storage?
        # Let's use a recursive helper.
        
        # To speed up, we can pre-compute valid differences? No, M is dynamic.
        
        # Heuristic: 
        # When choosing x for (r, j), we must ensure that for all i < j, (x - row[i]) is not in used_diffs.
        # Also, we must ensure that we CAN complete the rest of the row and subsequent rows.
        # A strong pruning rule: 
        # The number of available differences in [1, M] is M.
        # We have used |used_diffs| differences.
        # Remaining differences needed: T - |used_diffs|.
        # So M - |used_diffs| >= T - |used_diffs| => M >= T. (Already known)
        # More specific: For the current row, we need D - (j-1) more differences.
        
        def backtrack(r, j, current_row):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if j == k + 1:
                # Row complete
                rows.append(current_row)
                return backtrack(r + 1, 1, [0])
            
            # We are filling element j (1-indexed, so index j-1 in list, but list has 0 at index 0)
            # current_row has length j (indices 0 to j-1). We want to find value for index j.
            # Wait, if j=1, current_row=[0]. We want to find x_1.
            # If j=k, current_row has length k. We want to find x_k.
            
            last_val = current_row[-1]
            
            # Determine range for next value x
            # x > last_val
            # x <= M_limit
            # Also, x must be such that we can fit remaining (k - j) elements.
            # Min step is 1. So x + (k - j) <= M_limit => x <= M_limit - (k - j)
            
            max_x = M_limit - (k - j)
            
            # Iterate candidates
            # To improve search, we can try candidates in a randomized order?
            # Or just increasing? Increasing is usually better for packing.
            
            # Pruning: 
            # If we pick x, we add differences x - current_row[i] for i in 0..j-1.
            # If any is in used_diffs, skip.
            
            # We can also check if the NEW differences leave enough room for the rest of the row?
            # This is complex. Let's rely on the base case and simple conflict check.
            
            # Optimization: If we fail often, we might want to shuffle candidates.
            # But for "minimizing max", we want to find ANY valid solution within M_limit.
            
            for x in range(last_val + 1, max_x + 1):
                # Check conflicts
                ok = True
                new_diffs = []
                for i in range(len(current_row)):
                    d = x - current_row[i]
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if not ok:
                    continue
                
                # Tentatively add
                for d in new_diffs:
                    used_diffs.add(d)
                current_row.append(x)
                
                if backtrack(r, j + 1, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        # Start
        # We need to handle the first element specially? No, j starts at 1.
        # backtrack(0, 1, [0])
        
        # To allow multiple attempts within time_limit, we could restart with different random seeds?
        # Since the search is deterministic given the order, we can randomize the candidate order
        # or the row order.
        
        # Let's add a small randomization: shuffle the range of x?
        # No, that breaks the "minimize" aspect if we are just checking validity for a fixed M.
        # But for finding A solution, randomization helps avoid pathological cases.
        
        # However, simple increasing order is very strong for packing.
        
        return backtrack(0, 1, [0])

    # 3. Main Loop: Binary Search on M
    # We know M >= T.
    # We have an initial M_max from greedy.
    # We want the smallest M such that a solution exists.
    
    low = T
    high = current_best_max
    
    # If greedy is already optimal (M == T), we are done.
    if low == high:
        return best_rows
    
    # We will try to find a solution for M values between low and high.
    # Since the search space is huge, we might not be able to prove non-existence for many M.
    # Strategy:
    # 1. Try M = low. If found, return.
    # 2. Try M = low + 1, etc?
    # Given time constraints, we should prioritize finding a valid solution for a smaller M.
    
    # A more robust strategy for limited time:
    # Iterate M from low to high.
    # For each M, run the solver with a time budget.
    # If found, update best and try to go lower? No, we are going up.
    # If we find a solution for M, that's our answer (since we started from low).
    # If we run out of time, return the best found (which might be the greedy one if no improvement found).
    
    # Wait, the greedy solution might not be optimal.
    # We want to minimize M.
    # So we check M = low, low+1, ...
    
    best_found_rows = best_rows
    best_found_max = current_best_max
    
    # We can also try to improve the greedy solution by random restarts?
    # The backtracking is exact. If it finds a solution for M, it's valid.
    
    # Let's allocate time per M attempt.
    # Number of attempts possible: (deadline - now) / time_per_attempt
    
    # A smart approach: 
    # Try M = low. Give it a generous chunk of time.
    # If it fails, try M = low + 1, etc.
    # But if M=low is impossible, we waste time.
    
    # Alternative: 
    # Use the greedy solution's max as the upper bound.
    # Try to reduce it step by step?
    
    # Let's try a few specific M values close to the lower bound.
    
    # Heuristic: The gap between T and M is often small for these problems.
    
    attempts = 0
    max_attempts = 10 # Don't try too many values if each is expensive
    
    # We'll try M values: low, low+1, ..., up to high or until time runs out.
    # But we should be careful not to spend all time on M=low if it's impossible.
    
    # Let's use a time-decaying budget.
    # First attempt: 50% of remaining time?
    # Subsequent: 50% of remaining?
    
    # Actually, since we have a greedy solution, we can just try to beat it.
    # Target M = current_best_max - 1.
    # If we find a solution, update best and try M-1 again.
    # This is "Iterative Deepening" on the max value.
    
    target_max = current_best_max - 1
    
    while target_max >= low and time.time() < deadline - 5.0:
        # Allocate time for this attempt
        # If we are close to the lower bound, we might need more time?
        # Or less?
        # Let's just use a fixed slice, e.g., 5 seconds, or proportional to remaining.
        
        # Let's use 10% of remaining time, min 1s, max 20s.
        remaining = deadline - time.time()
        attempt_time = min(20.0, max(1.0, remaining * 0.1))
        
        # Run solver
        found = solve_with_limit(target_max, attempt_time)
        
        if found:
            # We found a valid solution with max <= target_max.
            # The solver returns True, but we need the rows.
            # My solver above doesn't return rows directly, it modifies local variables.
            # I need to capture the rows.
            
            # Let's refactor solve_with_limit to return the rows.
            
            # Re-implementing solve_with_limit to return rows
            # ... (See below in code)
            
            # For now, if found, we update best.
            # We need to extract the rows from the solver.
            
            # Let's assume the solver works and returns rows.
            # I will rewrite the solver to return the solution.
            
            # Break out if we hit the lower bound
            if target_max == low:
                break
            
            target_max -= 1
        else:
            # If we couldn't find a solution for target_max, we can't guarantee it's impossible.
            # But we can't afford to spend infinite time.
            # We should probably stop and return the best we have.
            # Or try a slightly higher M? No, we want to minimize.
            # If we fail for M, it's likely M is too small.
            # So we stop.
            break
            
    return best_found_rows

# Refactored Solver to return rows
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = k * (k + 1) // 2
    T = n_rows * D
    
    def greedy_solution():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                val = row[-1] + 1
                while True:
                    conflict = False
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        break
                    val += 1
                row.append(val)
                for i in range(len(row) - 1):
                    used_diffs.add(val - row[i])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = max(row[-1] for row in best_rows)
    
    if T == best_max:
        return best_rows
        
    def solve_for_max(M_limit, time_limit):
        """
        Returns a valid list of rows with max entry <= M_limit, or None.
        """
        used_diffs = set()
        rows = []
        
        # We use a list to hold the current solution state to avoid passing large objects
        # and to allow modification in place.
        
        def backtrack(r, j, current_row):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if j == k + 1:
                rows.append(current_row[:])
                return backtrack(r + 1, 1, [0])
            
            last_val = current_row[-1]
            max_x = M_limit - (k - j)
            
            if last_val + 1 > max_x:
                return False
            
            # Try candidates
            # To add some stochasticity and potentially escape bad paths faster,
            # we could shuffle a small window, but for packing, increasing is best.
            
            for x in range(last_val + 1, max_x + 1):
                ok = True
                new_diffs = []
                for i in range(len(current_row)):
                    d = x - current_row[i]
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if not ok:
                    continue
                
                for d in new_diffs:
                    used_diffs.add(d)
                current_row.append(x)
                
                if backtrack(r, j + 1, current_row):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        if backtrack(0, 1, [0]):
            return rows
        return None

    target_max = best_max - 1
    lower_bound = T
    
    while target_max >= lower_bound and time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        # Give the solver a reasonable chunk of time
        # If we are close to the limit, give it less.
        attempt_time = min(10.0, max(0.5, remaining * 0.2))
        
        sol = solve_for_max(target_max, attempt_time)
        
        if sol is not None:
            # Validate just in case
            # (Assuming solver is correct)
            best_rows = sol
            best_max = target_max
            if best_max == lower_bound:
                break
            target_max -= 1
        else:
            # If we failed, it might be due to time limit or infeasibility.
            # We stop to preserve the best known valid solution.
            break
            
    return best_rows
