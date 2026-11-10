import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def calculate_scope(rows):
        """Calculate the maximum entry across all rows."""
        return max(row[-1] for row in rows)
    
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
    
    def get_cost(rows):
        """Cost function: scope + penalty for conflicts (should be 0 if valid)."""
        valid, scope = is_valid_and_scope(rows)
        if valid is None:
            return float('inf')
        return scope
    
    def greedy_construct(rng, max_val_hint=200):
        """
        Randomized greedy construction.
        Returns a valid solution or None.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try random candidates from a range
                found = False
                # Bias towards smaller values
                for _attempt in range(100):
                    if time.time() > deadline:
                        return None
                    # Sample from a range, biased towards smaller
                    upper = min(max_val_hint, row[-1] + 50)
                    x = row[-1] + 1 + rng.randint(0, max(0, upper - row[-1] - 1))
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    # Try sequential search
                    x = row[-1] + 1
                    while x <= max_val_hint:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            row.append(x)
                            used_diffs.update(new_diffs)
                            found = True
                            break
                        x += 1
                
                if not found:
                    return None
            
            rows.append(row)
        
        return rows
    
    def local_search(rows, rng, time_limit):
        """
        Local search to minimize scope.
        Uses row regeneration and element swaps.
        """
        best_rows = [row[:] for row in rows]
        best_scope = calculate_scope(best_rows)
        current_rows = [row[:] for row in rows]
        current_scope = best_scope
        
        while time.time() < time_limit:
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.7:
                # Regenerate a random row
                row_idx = rng.randint(0, n_rows - 1)
                old_row = current_rows[row_idx][:]
                
                # Collect diffs used by other rows
                other_diffs = set()
                for i, row in enumerate(current_rows):
                    if i != row_idx:
                        other_diffs.update(get_row_diffs(row))
                
                # Regenerate row with constraint that max <= current_scope (or slightly less)
                target_max = best_scope
                if rng.random() < 0.3:
                    target_max = best_scope - rng.randint(1, 5)
                
                new_row = [0]
                success = True
                while len(new_row) < k + 1:
                    found = False
                    # Try random candidates
                    for _ in range(50):
                        if time.time() > time_limit:
                            return best_rows
                        upper = min(target_max, new_row[-1] + 30)
                        if upper <= new_row[-1]:
                            break
                        x = new_row[-1] + 1 + rng.randint(0, max(0, upper - new_row[-1] - 1))
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & other_diffs):
                            new_row.append(x)
                            found = True
                            break
                    
                    if not found:
                        # Sequential
                        x = new_row[-1] + 1
                        while x <= target_max:
                            new_diffs = {x - a for a in new_row}
                            if not (new_diffs & other_diffs):
                                new_row.append(x)
                                found = True
                                break
                            x += 1
                    
                    if not found:
                        success = False
                        break
                
                if success and len(new_row) == k + 1:
                    # Check if this reduces scope
                    new_scope = max(new_row[-1], max(r[-1] for i, r in enumerate(current_rows) if i != row_idx))
                    if new_scope <= current_scope:
                        current_rows[row_idx] = new_row
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best_rows = [row[:] for row in current_rows]
                            best_scope = new_scope
                else:
                    # Revert
                    current_rows[row_idx] = old_row
            
            else:
                # Try to reduce a specific element
                row_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)  # Position to try to reduce
                old_val = current_rows[row_idx][pos]
                
                # Collect diffs used by other rows
                other_diffs = set()
                for i, row in enumerate(current_rows):
                    if i != row_idx:
                        other_diffs.update(get_row_diffs(row))
                
                # Try to find a smaller value for this position
                new_val = old_val - 1
                found = False
                while new_val > current_rows[row_idx][pos-1]:
                    # Check if new_val creates conflicts
                    new_diffs = set()
                    for j in range(pos + 1):
                        if j != pos:
                            new_diffs.add(abs(new_val - current_rows[row_idx][j]))
                    
                    if not (new_diffs & other_diffs):
                        # Check internal consistency
                        valid_internal = True
                        for i in range(pos):
                            for j in range(i + 1, pos):
                                if i == pos or j == pos:
                                    continue
                                if current_rows[row_idx][j] - current_rows[row_idx][i] in new_diffs:
                                    valid_internal = False
                                    break
                            if not valid_internal:
                                break
                        
                        if valid_internal:
                            current_rows[row_idx][pos] = new_val
                            found = True
                            break
                    new_val -= 1
                
                if not found:
                    current_rows[row_idx][pos] = old_val
            
            # Occasionally restart with a new greedy solution
            if rng.random() < 0.05:
                new_sol = greedy_construct(rng, best_scope + 10)
                if new_sol:
                    new_scope = calculate_scope(new_sol)
                    if new_scope < current_scope:
                        current_rows = [row[:] for row in new_sol]
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best_rows = [row[:] for row in current_rows]
                            best_scope = new_scope
        
        return best_rows
    
    # Main algorithm
    best_solution = None
    best_scope = float('inf')
    
    rng = random.Random()
    
    # Phase 1: Try to find any valid solution with randomized greedy
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        sol = greedy_construct(rng, 200)
        if sol:
            valid, scope = is_valid_and_scope(sol)
            if valid:
                if scope < best_scope:
                    best_scope = scope
                    best_solution = sol
    
    if not best_solution:
        # Fallback: simple greedy
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
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        improved = local_search(best_solution, rng, deadline - 5)
        if improved:
            valid, scope = is_valid_and_scope(improved)
            if valid and scope < best_scope:
                best_scope = scope
                best_solution = improved
    
    return best_solution
