import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    # Quick greedy to get an initial upper bound
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Greedy smallest valid x
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    diffs = {x - a for a in row}
                    if not (diffs & used) and len(diffs) == len(row):
                        row.append(x)
                        used |= diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    # Get initial best
    best_rows = greedy_solution()
    best_scope = max(max(r) for r in best_rows)
    
    # Backtracking search to improve
    # We will try to find a solution with scope < best_scope
    # We can run this multiple times with different random seeds or perturbation if needed,
    # but standard backtracking with "try small x first" is deterministic.
    # To allow exploration, we can randomize the order of x candidates slightly or use restarts.
    
    def backtrack(row_idx, current_row, used_diffs):
        nonlocal best_scope, best_rows, deadline
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            # All rows constructed
            # Calculate scope
            scope = max(max(r) for r in best_rows) # Wait, best_rows is updated outside?
            # Actually, we should track the current solution's scope.
            # Let's pass current rows or calculate scope at the end.
            # Since we build rows one by one, we can keep a list of completed rows.
            pass # Placeholder, logic needs to track completed rows.

    # Let's restructure to track completed rows.
    
    def search(row_idx, current_row, used_diffs, completed_rows):
        nonlocal best_scope, best_rows, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_row) == k + 1:
            # Row complete
            # Check if we can prune based on current row's max?
            # If max(current_row) > best_scope, we might still be okay if other rows are small?
            # No, the global scope is max of all rows.
            # If max(current_row) > best_scope, this solution is worse than current best.
            # But we are trying to *find* a better one.
            # So if max(current_row) >= best_scope, we can prune?
            # Yes, because we want strictly smaller scope (or equal, but usually we want improvement).
            # If we find a solution with scope < best_scope, we update.
            # If max(current_row) >= best_scope, it cannot lead to a better solution.
            if max(current_row) >= best_scope:
                return
                
            # Add to used_diffs
            row_diffs = set()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    row_diffs.add(current_row[j] - current_row[i])
            
            new_used = used_diffs | row_diffs
            
            if row_idx == n_rows - 1:
                # Last row
                # We have a full solution
                # Scope is max of max of all rows
                # Since we pruned if any row >= best_scope, this new scope is < best_scope
                new_scope = max(max(r) for r in completed_rows + [current_row])
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = completed_rows + [current_row[:]]
                    # We can stop here if we want the first improvement,
                    # or continue to find even better.
                    # Given time limits, finding one good improvement is valuable.
                    # But maybe we can find 112? Unlikely quickly.
                    # Let's continue searching for a bit more to see if we can go lower.
            else:
                # Recurse to next row
                search(row_idx + 1, [0], new_used, completed_rows + [current_row[:]])
            return

        # Extend current_row
        # Heuristic: try small x first
        # Optimization: Precompute internal diffs?
        # For each candidate x, check diffs against used_diffs and internal diffs.
        
        # To speed up, we can maintain a set of internal diffs for current_row
        # But current_row changes.
        # Let's just compute on the fly. k is small (7).
        
        last_val = current_row[-1]
        # We can bound x.
        # If x - 0 > best_scope, then x > best_scope.
        # Since we prune if max(current_row) >= best_scope,
        # we should stop trying x if x >= best_scope.
        
        # Also, we need to ensure we don't exceed time.
        
        # To allow some exploration if deterministic fails or gets stuck,
        # we can add a small randomization or restarts.
        # But "try small first" is a strong heuristic for minimizing max.
        
        for x in range(last_val + 1, best_scope):
            # Check validity
            valid = True
            # Check against used_diffs
            # Check against internal diffs
            # Internal diffs of current_row
            # We can precompute internal diffs for current_row once per level?
            # No, current_row is fixed at this level.
            
            # Compute diffs with x
            new_diffs = {x - a for a in current_row}
            
            # Check intersection with used_diffs
            if new_diffs & used_diffs:
                continue
                
            # Check intersection with internal diffs of current_row
            # We need to check if any d in new_diffs is already in current_row's diffs.
            # current_row diffs:
            # Since k is small, we can just iterate.
            # Or maintain a set.
            # Let's maintain a set of internal diffs for the current row being built.
            # This requires passing it down.
            
            # For simplicity and correctness first:
            internal_diffs = set()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    internal_diffs.add(current_row[j] - current_row[i])
            
            if new_diffs & internal_diffs:
                continue
                
            # Valid
            search(row_idx, current_row + [x], used_diffs, completed_rows)
            
            # Early exit if we found a perfect solution (scope == lower bound)?
            # Lower bound is 112. If best_scope == 112, we can stop.
            if best_scope == 112:
                return

    # Start search
    # We might want to try a few different starting orders or random perturbations
    # if the first deterministic run doesn't improve much.
    # But let's try the deterministic "small x first" first.
    
    # To increase chance of finding a better solution than greedy,
    # we can also try randomizing the order of x candidates slightly.
    # However, the directive asks for a structural change to backtracking.
    # I will implement the backtracking as described.
    
    # One issue: The search space is huge.
    # Pruning `max(current_row) >= best_scope` is strong.
    # But we also need to prune based on "can we possibly finish the row within best_scope?"
    # If current_row[-1] + min_gaps * remaining_elements > best_scope, prune.
    # Min gap is 1.
    # So if current_row[-1] + (k - len(current_row) + 1) > best_scope, prune.
    
    # Let's add this pruning.
    
    def search_optimized(row_idx, current_row, used_diffs, completed_rows, internal_diffs):
        nonlocal best_scope, best_rows, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_row) == k + 1:
            if max(current_row) >= best_scope:
                return
            
            row_diffs = internal_diffs # Passed in
            
            new_used = used_diffs | row_diffs
            
            if row_idx == n_rows - 1:
                new_scope = max(max(r) for r in completed_rows + [current_row])
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = completed_rows + [current_row[:]]
            else:
                search_optimized(row_idx + 1, [0], new_used, completed_rows + [current_row[:]], set())
            return

        # Pruning: Can we fit the rest of the row within best_scope?
        # Current last element is current_row[-1].
        # We need (k + 1 - len(current_row)) more elements.
        # Min increment is 1.
        # So min possible max is current_row[-1] + (k + 1 - len(current_row)).
        if current_row[-1] + (k + 1 - len(current_row)) >= best_scope:
            return

        last_val = current_row[-1]
        
        # Try x from last_val + 1 to best_scope - 1
        # Note: x can be up to best_scope - 1 because if x == best_scope,
        # max(current_row) >= best_scope, which we prune.
        
        for x in range(last_val + 1, best_scope):
            new_diffs = {x - a for a in current_row}
            
            if new_diffs & used_diffs:
                continue
                
            if new_diffs & internal_diffs:
                continue
                
            # Update internal_diffs for recursion
            new_internal_diffs = internal_diffs | new_diffs
            
            search_optimized(row_idx, current_row + [x], used_diffs, completed_rows, new_internal_diffs)
            
            if best_scope == 112:
                return

    # Run search
    # We might need to restart or use different heuristics if it gets stuck.
    # But let's try this first.
    
    # To handle cases where the greedy solution is already optimal or very close,
    # and the search takes too long to find an improvement, we should ensure
    # we return the best found.
    
    # Initial state for search
    search_optimized(0, [0], set(), [], set())
    
    # If the search didn't improve, we return the greedy solution.
    # If it did, best_rows is updated.
    
    # Final validation
    def check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not check_valid(best_rows):
         # Fallback to a known valid structure if something went wrong
         # But best_rows came from greedy or valid backtracking, so should be valid.
         # If check fails, there's a bug.
         # Let's return the greedy one which is definitely valid.
         return greedy_solution()
         
    return best_rows
