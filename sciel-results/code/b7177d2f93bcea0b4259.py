import random
import time
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to check validity of a set of rows
    def is_valid(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    # Initial baseline using pure greedy
    def pure_greedy():
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

    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)

    # Strategy:
    # 1. Generate a pool of candidate rows (Golomb rulers) with scope <= current_best_max - 1.
    # 2. Build a conflict graph where nodes are candidate rows, edges if they share a difference.
    # 3. Use a Maximum Weight Independent Set (MWIS) solver to pick n_rows non-conflicting rows.
    #    To minimize the max scope, we can either:
    #    a) Filter candidates by scope and run MWIS, iterating down.
    #    b) Assign weights based on scope and find the independent set of size n_rows with min max weight.
    #    We will use a hybrid: generate candidates with scope <= limit, then use a branch-and-bound
    #    to find an independent set of size n_rows. If found, we can try to reduce the limit.

    def generate_random_golomb_row(max_val):
        """Generate a random Golomb ruler (row) with max entry <= max_val."""
        row = [0]
        for _ in range(k):
            current_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    current_diffs.add(row[j] - row[i])
            
            found = False
            # Randomize the order of selection to explore different parts of the space
            candidates = list(range(row[-1] + 1, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & current_diffs:
                    row.append(x)
                    found = True
                    break
            
            if not found:
                return None
            
        return row

    def solve_mwis(nodes, n_select, deadline):
        """
        Find an independent set of size n_select in the graph defined by nodes.
        nodes: list of (row, diffs_set, scope)
        Uses a simple branch and bound with randomization.
        Returns the list of rows if found, else None.
        """
        if not nodes:
            return None
        
        # Sort nodes by scope ascending to prefer smaller scopes early?
        # Or random order?
        # Let's sort by scope to bias towards smaller max.
        nodes_sorted = sorted(nodes, key=lambda x: x[2])
        
        best_set = None
        best_max_scope = float('inf')
        
        # We need to select exactly n_select nodes that are mutually non-conflicting.
        
        def backtrack(idx, selected, used_diffs, current_max_scope):
            nonlocal best_set, best_max_scope
            
            if time.time() > deadline:
                return
            
            if len(selected) == n_select:
                # Found a valid set
                if current_max_scope < best_max_scope:
                    best_max_scope = current_max_scope
                    best_set = selected[:]
                return
            
            # Pruning: if we can't reach n_select nodes even if we take all remaining, return
            remaining = len(nodes_sorted) - idx
            if len(selected) + remaining < n_select:
                return
            
            # Pruning: if current_max_scope is already >= best_max_scope, we might still want to continue
            # if we hope to find a better one? No, if we are minimizing max, and we already exceeded the best,
            # we can only improve if we find a set with max < best_max_scope.
            # But we are adding nodes. If current_max_scope >= best_max_scope, any superset will have max >= current_max_scope.
            # So if current_max_scope >= best_max_scope, we can prune?
            # Wait, we are trying to find ANY set with max < best_max_scope.
            # If current_max_scope >= best_max_scope, we cannot improve.
            if current_max_scope >= best_max_scope:
                return

            # Heuristic: try nodes with smaller scope first?
            # We iterate from idx to end.
            
            for i in range(idx, len(nodes_sorted)):
                row, diffs, scope = nodes_sorted[i]
                
                # Check conflict
                if diffs & used_diffs:
                    continue
                
                new_used = used_diffs | diffs
                new_max_scope = max(current_max_scope, scope)
                
                selected.append(row)
                backtrack(i + 1, selected, new_used, new_max_scope)
                selected.pop()
                
                # If we found a solution with scope 0 (impossible) or lower bound, we could break?
                # Lower bound for scope is at least k+1? No, 0 is min, max is at least k.
                # Actually, min max entry for k+1 marks is k (0..k).
                # But we are looking for distinct differences.
                # Greedy baseline is usually much larger.
                
                # Early termination if we found a very good solution?
                # If new_max_scope == lower_bound, we can stop.
                # lower_bound = n_rows * k*(k+1)//2 ? No, that's total diffs.
                # Max entry must be at least total_diffs.
                # Let's just let the time limit handle it.

        backtrack(0, [], set(), 0)
        
        return best_set

    # Main loop: Try to find a valid set of n_rows rows with max entry < best_max
    # We will maintain a pool of candidate rows.
    
    # To improve efficiency, we can try to build the set row by row.
    # Row 1: Pick a random Golomb row with max <= limit.
    # Row 2: Pick a random Golomb row with max <= limit that doesn't conflict with Row 1.
    # ...
    
    # This is a constraint satisfaction problem.
    # Let's use a randomized backtracking search for the set of rows.
    
    # We will iterate over scope limits.
    # Start from best_max - 1 and go down.
    # For each limit, generate a pool of candidates and run MWIS.
    
    # Since generating candidates is expensive, we do it in batches.
    
    # Let's try a few restarts with the current best_max - 1 limit first.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to find a solution with max < best_max
        # We use a time slice for each attempt
        time_slice = 5.0
        if time.time() + time_slice > deadline:
            time_slice = deadline - time.time()
            if time_slice < 0.1:
                break
        
        # Generate a pool of candidates with scope <= best_max - 1
        limit = best_max - 1
        pool_size = 100  # Number of candidates to generate
        candidates = []
        
        # Generate unique candidates
        seen_rows = set()
        for _ in range(pool_size):
            if time.time() > start_time + time_slice:
                break
            row = generate_random_golomb_row(limit)
            if row:
                # Check if it's a new row
                row_tuple = tuple(row)
                if row_tuple not in seen_rows:
                    seen_rows.add(row_tuple)
                    # Compute diffs
                    diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            diffs.add(row[j] - row[i])
                    candidates.append((row, diffs, row[-1]))
        
        if not candidates:
            continue
            
        # Run MWIS
        sol = solve_mwis(candidates, n_rows, start_time + time_slice)
        
        if sol:
            # Verify validity just in case
            if is_valid(sol):
                curr_max = get_max_entry(sol)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = [row[:] for row in sol]
        
        # If we have been trying for a while and not improving, 
        # we might want to try a looser limit to ensure we don't miss a better solution 
        # that requires a different structure?
        # No, if a solution with max M exists, it will be found with limit M.
        # If we fail to find it, it's a search failure.
        
        # To diversify, we can occasionally try to find a solution with limit = best_max 
        # (which should be easy) and then try to reduce it? 
        # But we already have a solution with best_max.
        
        # Let's add a small probability to try a limit slightly higher than best_max - 1 
        # to explore the neighborhood, in case the random generation is biased?
        # Actually, generate_random_golomb_row(limit) is constrained by limit.
        
        # One improvement: In find_set_with_limit, if we fail, we can try to 
        # reuse some rows from the previous best solution?
        # This is a form of hybrid.
        
        # For now, let's just keep running.
        
        # If we have spent a lot of time without improvement, we can break early?
        # No, keep going until deadline.
        
        # To prevent infinite loops if time_slice is very small, ensure progress.
        if time.time() > deadline:
            break

    return best_rows
