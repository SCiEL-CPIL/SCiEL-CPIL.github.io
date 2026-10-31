import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit (evaluator kills at 150, code runs in 120? Prompt says 120s limit in contract, 150s kill in directive. I will use 118s to be safe for the 120s contract)
    deadline = start_time + 118

    # Constants
    # We need to fit n_rows rows of k+1 elements.
    # Total differences = n_rows * k * (k+1) / 2.
    # Let M be the max value. Differences are in 1..M.
    # So M >= total_diffs.
    # Also, each row has k+1 elements in 0..M.
    
    # Lower bound on M:
    # 1. Counting: M >= n_rows * k * (k+1) / 2
    # 2. Packing: The smallest possible max element for a single row of size k+1 is k(k+1)/2 (0, 1, 3, 6... no, 0,1,2,3... max is k). 
    #    Wait, if we just pick 0,1,2,...,k, max is k. Diffs are 1..k.
    #    But diffs must be distinct across ALL rows.
    #    So we need n_rows * k(k+1)/2 distinct differences.
    #    The maximum difference in a row with max element M is M.
    #    So we need enough "slots" 1..M to hold all differences.
    #    Thus M >= n_rows * k(k+1)/2.
    
    # Let's start M at the theoretical lower bound and increase.
    base_M = n_rows * k * (k + 1) // 2
    
    # However, structural constraints might require M to be larger.
    # The "best known" for (4,7) is 134.
    # base_M for (4,7) is 4 * 7 * 8 / 2 = 112.
    # So we need to search M in [112, ...].
    
    # Strategy:
    # 1. Try to solve for M = base_M, base_M+1, ... up to a limit.
    # 2. For each M, use randomized backtracking with restarts.
    
    # To speed up backtracking, we precompute valid "shapes" or use a smart heuristic.
    # A row is a set of k+1 integers 0 = x_0 < x_1 < ... < x_k <= M.
    # The differences x_j - x_i must be unique globally.
    
    # We can represent the "used differences" as a boolean array of size M+1.
    
    def solve_for_M(M, time_limit):
        """
        Attempts to find n_rows valid rows with max element <= M.
        Returns list of rows if found, else None.
        """
        used_diffs = [False] * (M + 1)
        rows = []
        
        # Heuristic order for rows? 
        # Maybe rows with larger "density" of small numbers first?
        # Or just random order?
        # Let's try to construct rows one by one.
        
        # We need to pick a row from the available differences.
        # This is a constraint satisfaction problem.
        
        # Backtracking search state:
        # current_row: list of integers
        # row_index: which row we are building (0 to n_rows-1)
        
        # To make it efficient, we can pre-calculate all valid "prefixes" of a row?
        # No, that's too big.
        
        # Instead, we use a recursive function.
        # To avoid deep recursion overhead, we can use an iterative stack or just set recursion limit.
        sys.setrecursionlimit(10000)
        
        # Randomized restarts
        best_solution = None
        restarts = 0
        max_restarts = 1000  # Limit per M value
        
        while time.time() < time_limit and restarts < max_restarts:
            restarts += 1
            used_diffs = [False] * (M + 1)
            rows = []
            
            # Shuffle the "priority" of starting the search?
            # Actually, the order of elements in a row matters for the search tree.
            # We can randomize the choice of the next element.
            
            def backtrack(row_idx, current_row):
                if time.time() > time_limit:
                    return False
                
                if row_idx == n_rows:
                    # All rows constructed
                    # Check if valid (should be by construction)
                    # We need to return the solution
                    # Since we can't return easily from nested func, we use nonlocal
                    nonlocal best_solution
                    best_solution = [r[:] for r in rows]
                    return True
                
                if len(current_row) == k + 1:
                    # Row complete, move to next
                    rows.append(current_row)
                    if backtrack(row_idx + 1, [0]):
                        return True
                    rows.pop()
                    # Undo differences added by this row?
                    # We added them in the loop below. We need to remove them.
                    # Wait, the logic below adds diffs as we extend the row.
                    # So when we pop the row, we must remove its diffs from used_diffs.
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            used_diffs[d] = False
                    return False
                
                # Extend current_row
                last_val = current_row[-1]
                
                # What are the possible next values?
                # x > last_val
                # x <= M
                # Also, x - y must not be in used_diffs for any y in current_row.
                
                # Heuristic: Try smaller x first? Or random?
                # Smaller x keeps the row "tight", which might be good for packing.
                # But randomization helps escape local minima.
                
                # Generate candidates
                # To speed up, we can jump.
                # But k is small (7), so row length is 8.
                # Depth is 8. Width is M.
                # M is ~112-134.
                # 134^8 is huge, but pruning is strong.
                
                # Let's try candidates from last_val + 1 to M.
                # Optimization: If x - last_val is in used_diffs, skip.
                
                # Randomize the order of candidates to allow different structures
                # But bias towards smaller values to keep M low?
                # Actually, we are fixing M. We just need to find ANY solution.
                # So uniform random order might be better for diversity.
                
                candidates = []
                for x in range(last_val + 1, M + 1):
                    valid = True
                    for y in current_row:
                        d = x - y
                        if used_diffs[d]:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Shuffle candidates for randomization
                random.shuffle(candidates)
                
                # Try each candidate
                for x in candidates:
                    # Add x to row
                    current_row.append(x)
                    
                    # Mark differences
                    for y in current_row[:-1]:
                        used_diffs[x - y] = True
                    
                    if backtrack(row_idx, current_row):
                        return True
                    
                    # Undo
                    for y in current_row[:-1]:
                        used_diffs[x - y] = False
                    current_row.pop()
                
                return False
            
            # Start backtracking
            if backtrack(0, [0]):
                if best_solution:
                    return best_solution
        
        return None

    # Main loop: Try increasing M
    # Start from lower bound
    # We might need to go higher than base_M.
    # Let's try up to base_M + 50 or so, or until time runs out.
    
    best_rows = None
    current_M = base_M
    
    # It's possible that base_M is too tight.
    # We will iterate M.
    # To avoid spending too much time on an impossible M, we limit time per M.
    # If we find a solution, we return it.
    # If we don't, we increase M.
    
    # However, if we find a solution at M, it's the optimal M for that search path.
    # But maybe a smaller M was possible but we missed it due to randomization?
    # Unlikely if we run enough restarts.
    # But to be safe, we could store the best solution found so far (smallest M).
    
    # Since the score is the largest entry, finding the smallest M is the goal.
    # So we should start at base_M and go up.
    # The first M for which we find a solution is likely the minimum (or close).
    
    # Let's allocate time.
    # If we find a solution quickly at base_M, great.
    # If not, we might need to try base_M+1, etc.
    
    # Strategy:
    # Try M = base_M with a time budget.
    # If fail, try M = base_M + 1 with a time budget.
    # ...
    # Keep the best solution found (smallest M).
    
    # Since (4,7) best known is 134 and bound is 112, the gap is 22.
    # We should be able to cover this range.
    
    # Time budget per M:
    # Start with 5s, increase if failing?
    # Or just distribute remaining time?
    # Let's try 5s per M initially.
    
    time_per_M = 5.0
    
    while time.time() < deadline - 2:
        t_start = time.time()
        sol = solve_for_M(current_M, t_start + time_per_M)
        
        if sol:
            # Found a solution with max element <= current_M
            # Actually, the max element in the rows might be less than current_M.
            # We should check the actual max.
            actual_max = max(max(r) for r in sol)
            
            if best_rows is None or actual_max < max(max(r) for r in best_rows):
                best_rows = sol
            
            # If we found a solution at the lower bound, it's likely optimal.
            # But maybe we can do better? No, base_M is the counting bound.
            # If actual_max == base_M, we are optimal.
            if actual_max == base_M:
                break
            
            # If we found a solution, we can try to improve it?
            # Or just accept it?
            # Given the time constraints, accepting the first valid solution for the lowest M
            # that yields a result is a good heuristic.
            # However, it's possible that a solution exists at base_M but we failed to find it
            # within time_per_M, and we found one at base_M+1.
            # So we should continue trying lower M's? No, we are going up.
            # So if we found one at M, we know M is feasible.
            # We don't know if M-1 is feasible.
            # But we tried M-1 and failed (or didn't try it if we started at base_M).
            # So the current best is the smallest M we successfully solved.
            
            # Should we stop?
            # If we are at base_M, we stop.
            # If we are at base_M + delta, maybe base_M + delta - 1 is solvable but we ran out of time?
            # To be robust, we could do a final check on the best M found minus 1?
            # But that costs time.
            
            # For now, let's assume the first success is the best we can get in this run.
            # We can try to polish it?
            # The task asks for the set. The score is the max entry.
            # Returning the found set is valid.
            
            break
        
        # If no solution found, increase M
        current_M += 1
        # Maybe increase time budget for harder M?
        # time_per_M = min(time_per_M * 1.2, 10.0)
        
        # If current_M gets too high, we might run out of time.
        # Cap the search space?
        if current_M > base_M + 100:
            break

    # Fallback: If no solution found via backtracking, use the greedy+local search from the original?
    # Or just return a valid (though large) solution.
    # We must ensure we return a VALID solution.
    # If best_rows is None, we need a fallback.
    
    if best_rows is None:
        # Fallback to greedy construction (which is always valid, just large)
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
        best_rows = rows

    return best_rows
