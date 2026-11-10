import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Estimate a reasonable upper bound for the max difference
    # Simple counting: need total_diffs_needed distinct differences
    # If max entry is M, differences range from 1 to M
    # So M >= total_diffs_needed
    # But in practice we need more room
    est_upper = max(total_diffs_needed + 10, 150)
    
    def solve_with_backtracking(rng, max_scope, time_limit):
        """
        Randomized backtracking with bitmask difference tracking.
        Tries to build rows one by one, using backtracking within each row.
        Returns a valid solution or None.
        """
        # Track used differences with a set for fast lookup
        used_diffs = set()
        rows = []
        
        def backtrack_row(current_row, target_max):
            """
            Build a row starting with current_row (already has 0 and possibly more).
            Returns the completed row or None if stuck.
            """
            if len(current_row) == k + 1:
                return current_row
            
            # Check time
            if time.time() > time_limit:
                return None
            
            last = current_row[-1]
            
            # Generate candidate values in randomized order
            # We want values > last and <= target_max
            min_val = last + 1
            max_val = target_max
            
            if min_val > max_val:
                return None
            
            # Create a list of candidates and shuffle for randomization
            # But bias towards smaller values
            candidates = []
            # First add a few small candidates
            for v in range(min_val, min(min_val + 15, max_val + 1)):
                candidates.append(v)
            # Then add random candidates
            remaining = list(range(min_val, max_val + 1))
            # Remove the ones we already added
            added_set = set(candidates)
            remaining = [x for x in remaining if x not in added_set]
            rng.shuffle(remaining)
            candidates.extend(remaining[:min(30, len(remaining))])
            
            # Shuffle the initial small ones too for some randomization
            rng.shuffle(candidates[:min(15, len(candidates))])
            
            for x in candidates:
                if time.time() > time_limit:
                    return None
                
                # Check if adding x creates any conflicts
                conflict = False
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Try this value
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                result = backtrack_row(current_row, target_max)
                
                if result is not None:
                    return result
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        # Build rows one by one
        for r in range(n_rows):
            if time.time() > time_limit:
                return None
            
            # Try to build this row with backtracking
            result = backtrack_row([0], max_scope)
            if result is None:
                return None
            rows.append(result)
        
        return rows
    
    def greedy_construct(rng, max_val_hint):
        """
        Fast greedy construction as a fallback/initial solution.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                # Try values in order with some randomization
                base = row[-1] + 1
                # Try a window of values
                for offset in range(0, 80):
                    if time.time() > deadline:
                        return None
                    x = base + offset
                    if x > max_val_hint:
                        break
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    return None
            
            rows.append(row)
        
        return rows
    
    def is_valid_and_scope(rows):
        """Check if solution is valid and return scope."""
        used = set()
        max_val = 0
        for row in rows:
            if len(row) != k + 1:
                return None, -1
            if row[0] != 0:
                return None, -1
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return None, -1
                    used.add(d)
            if row[-1] > max_val:
                max_val = row[-1]
        return rows, max_val
    
    rng = random.Random()
    
    # Phase 1: Quick greedy to get an initial solution
    best_solution = None
    best_scope = float('inf')
    
    for attempt in range(10):
        if time.time() > deadline - 60:
            break
        sol = greedy_construct(rng, est_upper)
        if sol:
            valid, scope = is_valid_and_scope(sol)
            if valid:
                if scope < best_scope:
                    best_scope = scope
                    best_solution = sol
                break  # Greedy is fast, take the first valid one
    
    if not best_solution:
        # Fallback: simple sequential greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows
    
    # Phase 2: Randomized backtracking to improve
    # Try to find solutions with progressively smaller max_scope
    current_target = best_scope - 1
    
    # Budget for backtracking attempts
    bt_time_budget = deadline - 10
    
    while time.time() < bt_time_budget and current_target >= total_diffs_needed:
        # Each backtracking attempt gets a time slice
        slice_time = min(5.0, bt_time_budget - time.time())
        
        if slice_time < 0.5:
            break
        
        sol = solve_with_backtracking(rng, current_target, time.time() + slice_time)
        
        if sol:
            valid, scope = is_valid_and_scope(sol)
            if valid and scope < best_scope:
                best_scope = scope
                best_solution = sol
                # We found something better; try to go even lower
                current_target = best_scope - 1
            else:
                # Didn't improve, try a slightly larger target next time
                # But still below current best
                if current_target >= best_scope:
                    current_target = best_scope - 1
                else:
                    # Stuck, try a random target below best
                    current_target = max(total_diffs_needed, 
                                         best_scope - rng.randint(1, 10))
        else:
            # Backtracking failed, try a larger target
            current_target = min(best_scope - 1, current_target + 3)
            if current_target >= best_scope:
                current_target = best_scope - 1
    
    return best_solution
