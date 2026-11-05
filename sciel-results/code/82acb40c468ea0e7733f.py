import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses a randomized backtracking search with forward checking.
    """
    start_time = time.time()
    max_time = 115.0
    
    best_rows = None
    best_scope = float('inf')
    
    # We will try to solve the problem by building rows one by one.
    # To improve diversity, we can shuffle the order in which we attempt to place elements
    # or use different random seeds for candidate ordering.
    
    # A simple but effective heuristic for difference sets is to keep values small.
    # We will use a recursive backtracking function.
    
    def solve():
        nonlocal best_rows, best_scope, start_time, max_time
        
        used_diffs = set()
        rows = []
        
        # We'll use a stack-based approach or recursion. Recursion is cleaner for small depth.
        # Depth is n_rows * k, which is 7 * 5 = 35 levels deep max. This is fine.
        
        def backtrack(row_idx, current_row, current_max_val):
            nonlocal best_rows, best_scope, start_time, max_time
            
            if time.time() - start_time > max_time:
                return False
            
            # Check if we have completed all rows
            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                return True
            
            # If we are starting a new row
            if len(current_row) == 0:
                # We can try to start the row with 0 (always)
                # The rest of the elements are chosen.
                # Actually, the first element is always 0.
                # So we call backtrack for the next element in this row.
                return backtrack(row_idx, [0], 0)
            
            # If we have completed the current row
            if len(current_row) == k + 1:
                # Commit this row
                rows.append(current_row)
                # Move to next row
                success = backtrack(row_idx + 1, [], 0)
                rows.pop()
                return success
            
            # Determine the next element in the current row
            last_val = current_row[-1]
            
            # Heuristic: Try candidates in a specific order.
            # We want to keep the scope small, so we prefer smaller values.
            # But we must ensure no difference conflicts.
            
            # Generate candidates. 
            # Upper bound for next value:
            # It must be > last_val.
            # Also, to potentially beat best_scope, the final value of this row must be < best_scope.
            # Since values are increasing, the last value is the max.
            # If we are at the last position (len == k), the candidate itself becomes the max.
            # If we are not at the last position, the candidate must allow room for k - len(current_row) more elements.
            
            # Let's just iterate from last_val + 1 upwards.
            # To avoid infinite loops or very large numbers, we can cap the search.
            # A reasonable cap: best_scope (if we want to improve) or some large number.
            # If best_scope is inf, we just find *a* solution.
            
            # Optimization: If we have a best_scope, we only care about solutions with scope < best_scope.
            # So the maximum value in any row should be < best_scope.
            
            upper_bound = best_scope - 1 if best_scope != float('inf') else 200 # Arbitrary high limit if no best found yet
            
            # If we are at the last element, the candidate must be < best_scope (if we want to improve)
            # Actually, if we find a solution with scope S, we update best_scope.
            # To find strictly better solutions, we need max_val < best_scope.
            
            # Let's try candidates in a randomized order biased towards smaller values?
            # Or just standard ascending order for determinism in the first few attempts,
            # then randomize for diversity?
            
            # Strategy: 
            # 1. Try ascending order (greedy-like) but with backtracking.
            # 2. If that fails or times out, try randomized orders.
            
            # For now, let's use a randomized shuffle of a range of candidates.
            # This helps escape local minima.
            
            # The range of candidates: last_val + 1 to min(upper_bound, last_val + 50)
            # Why +50? Usually differences don't jump too much in dense sets.
            # But for safety, if we are at the last element, we can go up to upper_bound.
            
            candidates = []
            max_candidate = upper_bound
            if len(current_row) < k:
                # We need to leave room for more elements.
                # Minimal increase is 1 per step.
                # So max_candidate should be upper_bound - (k - len(current_row))
                max_candidate = min(upper_bound, last_val + 100) # Heuristic window
            
            # Actually, let's just scan from last_val + 1 to max_candidate.
            # To add randomness, we can shuffle a subset or use a random start?
            # Better: Just iterate, but if we fail, the backtracking handles it.
            # To get different solutions, we can randomize the order of *rows*? 
            # No, rows are symmetric. 
            # We can randomize the choice of candidates.
            
            # Let's try a mix:
            # In early attempts, use ascending.
            # In later attempts, use randomized.
            
            # Simpler: Just use ascending order with backtracking. 
            # This is a depth-first search. It might get stuck in a deep branch.
            # To mitigate, we can use a timeout and restart with a different random seed for candidate selection.
            
            # Let's implement a simple randomized candidate selection.
            # We generate a list of valid candidates (no diff conflict) and pick one randomly?
            # No, that's too random. 
            # Let's stick to ascending order but allow the global search to restart.
            
            # Wait, the current seed program uses random restarts. 
            # My backtracker should also support restarts with different randomizations.
            
            # Let's modify the outer loop to do restarts.
            # Inside backtrack, we iterate candidates in ascending order.
            # This is deterministic for a given state. 
            # To get different paths, we need to change the state or the heuristic.
            
            # Idea: Randomly shuffle the *rows*? No, they are identical.
            # Idea: Randomly perturb the candidate order.
            
            # Let's try this: 
            # For each call to backtrack, we generate valid candidates.
            # We sort them by value.
            # With some probability, we swap adjacent elements to introduce noise?
            # Or better: We can randomize the *seed* of the random number generator used for tie-breaking?
            
            # Actually, for difference sets, ascending order is a very strong heuristic.
            # The main issue with greedy is it doesn't backtrack.
            # Backtracking with ascending order should find the lexicographically smallest solution, which is often optimal or near-optimal for scope minimization.
            
            # Let's try pure ascending backtracking first.
            
            for cand in range(last_val + 1, max_candidate + 1):
                if time.time() - start_time > max_time:
                    return False
                
                # Check differences
                new_diffs = {cand - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Check if this choice allows completion? (Forward Checking)
                # Simple check: Are there enough unused differences for the rest of the row?
                # This is complex. Let's just try it.
                
                # Tentatively add
                current_row.append(cand)
                used_diffs.update(new_diffs)
                
                # Recurse
                if backtrack(row_idx, current_row, cand):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        return backtrack(0, [], 0)

    # Run multiple attempts with different random seeds to explore different branches
    # Although ascending backtracking is deterministic, we can vary the "max_candidate" window
    # or add randomization to the candidate selection to escape deep dead-ends.
    
    # To make it non-deterministic and explore different parts of the space:
    # We can randomize the order of candidates slightly.
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Reset solver state
        # We need to re-implement or reset the closure variables.
        # It's easier to define the solver inside the loop or pass parameters.
        
        # Let's refactor to allow randomization of candidate order.
        
        used_diffs = set()
        rows = []
        
        def backtrack_randomized(row_idx, current_row, rng):
            nonlocal best_rows, best_scope, start_time, max_time
            
            if time.time() - start_time > max_time:
                return False
            
            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                return True
            
            if len(current_row) == 0:
                return backtrack_randomized(row_idx, [0], rng)
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                success = backtrack_randomized(row_idx + 1, [], rng)
                rows.pop()
                return success
            
            last_val = current_row[-1]
            upper_bound = best_scope - 1 if best_scope != float('inf') else 150
            
            # Determine candidate range
            # If we are at the last element, we can go up to upper_bound.
            # Otherwise, we should be conservative to leave room.
            # Let's use a window.
            window_size = 100
            max_candidate = min(upper_bound, last_val + window_size)
            
            candidates = list(range(last_val + 1, max_candidate + 1))
            
            # Filter valid candidates
            valid_cands = []
            for cand in candidates:
                new_diffs = {cand - a for a in current_row}
                if not (new_diffs & used_diffs):
                    valid_cands.append((cand, new_diffs))
            
            if not valid_cands:
                return False
            
            # Sort by value (ascending) but with random perturbation?
            # Or just shuffle?
            # Shuffling completely destroys the "small value" heuristic.
            # Let's sort by value, but if we have many valid candidates, 
            # we can pick a random one from the top N?
            
            # Strategy: Sort by value. 
            # To introduce randomness, we can swap the first few elements randomly.
            # Or, we can iterate in sorted order, which is the standard DFS.
            # If we want diversity, we can randomly select a starting index?
            
            # Let's stick to sorted order for efficiency, but rely on the "restart" 
            # mechanism to try different *global* configurations?
            # Actually, if the search is deterministic, all restarts will do the same thing.
            # So we MUST introduce randomness in the candidate selection.
            
            # Let's shuffle the valid candidates, but bias towards smaller values.
            # We can assign a weight = 1/value and sample?
            # Simpler: Sort by value, then randomly swap adjacent pairs with probability 0.5?
            # This keeps the order mostly ascending but allows jumps.
            
            # Let's just shuffle. It's a random search.
            # With backtracking, it will eventually find a solution if one exists.
            # The "best_scope" pruning helps.
            
            rng.shuffle(valid_cands)
            
            # But pure shuffle might pick large values early, increasing scope.
            # We want to minimize scope.
            # So we should prefer small values.
            
            # Compromise: Sort by value. Then, for the first few candidates, 
            # we can randomly pick one?
            
            # Let's try: Sort by value. Then, with probability 0.2, pick a random candidate 
            # from the top 10. Otherwise, pick the smallest.
            
            # Actually, let's just sort by value. 
            # To get different solutions, we can vary the `upper_bound` or `window_size`?
            # No, that changes the problem space.
            
            # Let's use a "Randomized Greedy" inside the backtrack?
            # No, backtrack tries all.
            
            # Okay, let's just shuffle. It's the most robust way to explore the space.
            # The "minimize scope" is handled by the global best tracking.
            # If we find a solution with scope 120, we update best_scope.
            # Subsequent searches will prune any branch where max_val > 120.
            # This focuses the search on better solutions.
            
            for cand, new_diffs in valid_cands:
                # Pruning: If this candidate is already too large to beat best_scope?
                # If len(current_row) == k (last element), and cand >= best_scope, skip.
                if len(current_row) == k and cand >= best_scope:
                    continue
                
                current_row.append(cand)
                used_diffs.update(new_diffs)
                
                if backtrack_randomized(row_idx, current_row, rng):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        rng = random.Random(attempt)
        backtrack_randomized(0, [], rng)

    if best_rows is None:
        # Fallback: Pure greedy (should not happen if backtracker works)
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
    
    return best_rows
