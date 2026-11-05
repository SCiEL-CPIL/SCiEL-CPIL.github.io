import time
import random
import sys
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Backtracking search with forward checking and MRV heuristic.
    We view the problem as assigning values to positions (row, col) such that
    all pairwise differences are unique.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Lower bound for scope (counting bound)
    # We need total_diffs_needed distinct differences.
    # The smallest distinct positive differences are 1, 2, ..., total_diffs_needed.
    # However, they must fit into rows of length num_elements.
    # A simple lower bound is total_diffs_needed, but structure matters.
    # For (7, 5), total_diffs = 7 * 15 = 105.
    # The known best is 112.
    
    lower_bound = total_diffs_needed
    # Heuristic upper bound to start
    # Start slightly above lower bound to find a valid solution quickly, then optimize.
    # Actually, we can do a binary search on the scope M, or just iterative deepening.
    # Given the time limit, we will try to find a solution with M close to the lower bound.
    
    # Data structures for backtracking
    # rows[i] = [0, x1, x2, ..., xk]
    # We assign values to positions (i, j) for i in 0..n_rows-1, j in 1..k.
    # Constraint: All differences rows[i][j] - rows[i][l] (j>l) must be unique globally.
    
    # We use a recursive backtracking that assigns elements one by one.
    # To make it efficient, we process rows one by one, and within a row, columns one by one.
    # Forward checking: Maintain a set of used differences. When placing a new element,
    # check if it creates any difference already used.
    
    # Optimization: Pre-compute pair indices for a row
    # Pairs in a row of length L: (0,1), (0,2), ..., (L-1, L-2)
    # When we place the j-th element (index j, value v), it creates differences
    # with all previous elements in the row: v - row[0], v - row[1], ..., v - row[j-1]
    
    # State:
    # - current_rows: list of lists
    # - used_diffs: set of all differences used so far
    
    # Heuristic: MRV is hard to implement dynamically in simple recursion without heavy overhead.
    # Instead, we use a "lookahead" or "fail-first" strategy:
    # 1. Try to fill rows sequentially.
    # 2. Within a row, the choice of the next element is constrained by the previous ones.
    # 3. We can randomize the order of rows or the search order to escape local minima.
    
    # Let's implement a randomized backtracking with a time limit.
    # We will try multiple restarts with different random seeds/orderings if time permits.
    
    best_solution = None
    best_scope = float('inf')
    
    # Helper to check if a partial row can be extended
    # Actually, we just check conflicts with used_diffs.
    
    def backtrack(row_idx, col_idx, current_rows, used_diffs, current_scope, max_scope):
        nonlocal best_solution, best_scope, start_time
        
        # Time check
        if time.time() - start_time > DEADLINE:
            return False
            
        # Pruning: If current_scope + remaining_min_increase > best_scope, prune?
        # Hard to estimate remaining_min_increase.
        
        if row_idx == n_rows:
            # All rows constructed
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [r[:] for r in current_rows]
            return True # We can return True to stop further search in this branch if we want, 
                        # but usually we want to find the MINIMUM, so we continue searching 
                        # unless we hit the lower bound.
                        # If current_scope == lower_bound, we can return True and stop all.
            # For now, let's just record and return True to allow backtracking to find other solutions
            # if we are doing a full search. But with time limits, we just want *a* good one.
            # If we find a solution, we record it. We don't necessarily stop immediately 
            # unless we are confident it's optimal or time is low.
            
        # If we are at the start of a new row
        if col_idx == 1:
            current_rows[row_idx] = [0]
            
        # Select next value for current_rows[row_idx][col_idx]
        # The value must be > current_rows[row_idx][col_idx - 1]
        # And < max_scope (we are trying to keep scope small)
        
        # Heuristic: Try values that minimize the maximum difference?
        # Or just iterate from smallest feasible to largest?
        # To find small scopes, we should try smaller values first?
        # But smaller values might lead to dead ends later.
        
        # Let's use a randomized order of candidate values, biased towards smaller values
        # to encourage compactness, but with randomness to avoid getting stuck.
        
        last_val = current_rows[row_idx][col_idx - 1]
        
        # Upper bound for this value:
        # It must be <= max_scope.
        # Also, the remaining elements in this row must fit.
        # Min gap is 1. So max possible value for last element is max_scope.
        # This element is at index col_idx. There are (num_elements - col_idx) elements left.
        # So value + (num_elements - col_idx) <= max_scope => value <= max_scope - (num_elements - col_idx)
        
        max_val_for_this_pos = max_scope - (num_elements - col_idx)
        
        # Candidates: last_val + 1 to max_val_for_this_pos
        # We also need to ensure that the differences created are not in used_diffs.
        
        # To speed up, we can pre-filter candidates?
        # For each candidate v, diffs = [v - current_rows[row_idx][i] for i in range(col_idx)]
        # If any diff in used_diffs, skip.
        
        # Randomization strategy:
        # Shuffle the range, but bias towards smaller values?
        # Or just pick a random valid candidate?
        # Let's try a few random valid candidates at each step to diversify.
        
        # For efficiency in backtracking, we iterate through candidates.
        # To avoid deep recursion on bad paths, we can randomize the start point.
        
        # Let's generate candidates in a randomized order, but constrained to be increasing.
        # Actually, since the row must be increasing, we must pick v > last_val.
        
        # Optimization: If we are close to the deadline, we should be more aggressive with pruning or stop.
        
        candidates = list(range(last_val + 1, max_val_for_this_pos + 1))
        if not candidates:
            return False
            
        # Shuffle to explore different branches
        random.shuffle(candidates)
        
        # To encourage small scopes, we could sort by value but add noise?
        # Let's just shuffle. The "max_scope" parameter acts as the constraint.
        # We will call this function with increasing max_scope values (Iterative Deepening).
        
        for v in candidates:
            # Check conflicts
            new_diffs = []
            valid = True
            for i in range(col_idx):
                d = v - current_rows[row_idx][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Place v
            current_rows[row_idx].append(v)
            used_diffs.update(new_diffs)
            
            # Recurse
            if col_idx + 1 == num_elements:
                # Row complete
                success = backtrack(row_idx + 1, 1, current_rows, used_diffs, max(v, current_scope), max_scope)
            else:
                success = backtrack(row_idx, col_idx + 1, current_rows, used_diffs, max(v, current_scope), max_scope)
                
            if success:
                # If we found a solution, we can return True to stop this branch
                # But we want the MINIMUM scope.
                # If we are doing iterative deepening, we stop when we find a solution for a given M.
                # So we return True.
                return True
            
            # Backtrack
            current_rows[row_idx].pop()
            for d in new_diffs:
                used_diffs.discard(d)
                
        return False

    # Iterative Deepening DFS (IDDFS)
    # Start with M = lower_bound, increment M until we find a solution or time runs out.
    
    # To avoid getting stuck in a single bad deep dive, we can use a timeout per M level?
    # Or just one global timeout.
    # Since IDDFS restarts search for each M, it might waste time if M is too low and infeasible.
    # For (7,5), LB=105. Best=112.
    # Let's try M from LB to LB + 20.
    
    # To add diversity, we can randomize the search order inside backtrack (already done via shuffle).
    # But IDDFS is deterministic in structure.
    
    # Let's add a "random restart" mechanism:
    # If we don't find a solution for a specific M within a short time slice, we give up on that M?
    # No, if it's infeasible, we must give up. If it's feasible but hard, we need time.
    
    # Strategy:
    # 1. Try M = LB. Limit time to 5s. If fail, move to next M.
    # 2. Try M = LB+1. Limit time to 10s.
    # 3. ...
    # This ensures we don't waste all time on an infeasible M.
    
    # However, simple time slicing in recursion is tricky.
    # Alternative: Randomized Restart.
    # Run backtracking with a fixed M (e.g., LB + 5) for a short time. If no solution, restart with new random seed.
    # If we find a solution, record it. Then try to improve it?
    # Or just keep trying with smaller M?
    
    # Let's do this:
    # For M in range(LB, LB + 30):
    #   Try to find a solution with scope <= M.
    #   Use a time budget of (total_time / 30) * (1 + M/LB) or similar.
    #   If found, update best and break (since we are increasing M, first found is best for this M).
    #   Actually, if we find a solution for M, we can stop if M == LB.
    #   If M > LB, we found a valid solution with scope <= M. But maybe there is one with scope < M?
    #   IDDFS guarantees the FIRST solution found at the smallest M is the optimal one (if the search is complete).
    #   But our search is randomized and time-limited, so it's not complete.
    
    # Refined Strategy:
    # 1. Start M = LB.
    # 2. Run randomized backtracking for a fixed duration (e.g., 10s).
    # 3. If solution found with scope S, update best.
    # 4. If no solution, increase M.
    # 5. Stop when time is up.
    
    # To make backtracking more effective, we can use "Value Ordering" heuristic:
    # Prefer values that leave more options for future steps? (Hard to compute).
    # Prefer values that are "central"?
    
    # Let's implement the IDDFS with randomized restarts within each M level.
    
    current_M = lower_bound
    max_M_to_try = lower_bound + 50  # Don't go too high
    
    while current_M <= max_M_to_try and time.time() - start_time < DEADLINE:
        # Time budget for this M
        # If we are close to deadline, give less time to higher M?
        # Actually, higher M is more likely to have a solution.
        # Let's give each M a chance.
        
        # Simple heuristic: 20% of remaining time per M, capped at 10s?
        remaining_time = DEADLINE - (time.time() - start_time)
        time_budget = min(5.0, remaining_time * 0.1) # 5 seconds per M level max
        
        if time_budget <= 0:
            break
            
        # Try to find a solution with scope <= current_M
        # We run backtracking with this constraint.
        # To diversify, we run multiple short runs if we don't find it in the first run?
        # Or just one long run with randomization inside?
        # One long run with randomization inside is better as it maintains state.
        
        # Reset state
        initial_rows = [[0] for _ in range(n_rows)]
        initial_used = set()
        
        # Run backtracking
        # We need to pass the time limit to the recursive function or check it there.
        # The global time check in backtrack() handles this.
        
        # To allow multiple restarts if the first one gets stuck in a deep bad path:
        # We can interrupt the backtrack after a short sub-limit?
        # Let's just run it. The randomization in candidates helps.
        
        success = backtrack(0, 1, initial_rows, initial_used, 0, current_M)
        
        if success and best_solution is not None:
            # Found a solution with scope <= current_M
            # Since we start from LB, this is the best possible for this M.
            # If current_M == lower_bound, we can't do better.
            if current_M == lower_bound:
                break
            # Otherwise, we have a solution. Can we find a better one with smaller M?
            # No, we started from LB and went up. So this is the first M that yielded a solution.
            # Due to randomized search, it's possible we missed a solution at LB, found one at LB+1.
            # To be safe, we could go back down? No, time.
            # We accept this solution.
            break
        else:
            # No solution found for this M within time budget
            current_M += 1
            
    if best_solution is None:
        # Fallback: Greedy construction with a large M
        # This ensures we return a valid result
        M = lower_bound + 100
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            for j in range(k):
                last = row[-1]
                # Try to find a valid next element
                # Greedy: pick smallest valid
                found = False
                for v in range(last + 1, M + 1):
                    diffs = [v - x for x in row]
                    if not any(d in used for d in diffs):
                        row.append(v)
                        used.update(diffs)
                        found = True
                        break
                if not found:
                    # Should not happen with large M
                    row.append(M) # Emergency
            rows.append(row)
        best_solution = rows
        best_scope = max(max(r) for r in rows)

    return best_solution

# EVOLVE-BLOCK-END
