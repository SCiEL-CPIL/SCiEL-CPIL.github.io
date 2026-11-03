import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    # Compute initial used differences mask
    # Max possible difference is bounded by the scope. 
    # We need to know the max scope to size the mask.
    # Let's use a set for simplicity first, then switch to int mask if needed.
    # Given k=7, max scope ~134. Total diffs ~ 4*28=112.
    # An integer bitmask is efficient.
    
    def get_used_mask(solution):
        mask = 0
        for row in solution:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d > 0:
                        mask |= (1 << d)
        return mask

    best_mask = get_used_mask(best_solution)
    best_scope = max(max(row) for row in best_solution)
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    if best_scope <= lower_bound:
        return best_solution

    rng = random.Random(42)

    # 2. Local Search with Scope Reduction
    while time.time() < deadline - 2.0:
        improved = False
        
        # Try to reduce scope by regenerating the row with the largest entry
        # Find the row with the largest max entry
        max_idx = max(range(n_rows), key=lambda i: best_solution[i][-1])
        current_row_max = best_solution[max_idx][-1]
        
        # Target a slightly smaller scope
        # Try to fit the row in [0, current_row_max - 1]
        # But also check if global scope can be reduced
        global_max = best_scope
        
        # Try to regenerate the worst row to fit in global_max - 1
        # Or just try to compact it.
        
        # Strategy: Try to regenerate the row with the largest max entry 
        # constrained by the current used_mask (excluding the row's own diffs)
        # and a target max value.
        
        # First, remove the current row's diffs from the mask
        row_diffs = set()
        row = best_solution[max_idx]
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                row_diffs.add(row[j] - row[i])
        
        # Create a temporary mask without this row's diffs
        # It's easier to just use a set for the "forbidden" diffs from other rows
        other_diffs = set()
        for r_idx in range(n_rows):
            if r_idx == max_idx:
                continue
            r = best_solution[r_idx]
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    other_diffs.add(r[j] - r[i])
        
        # Try to construct a new row for max_idx with max value < best_scope
        # Start from best_scope - 1 down to lower_bound
        target_max = best_scope - 1
        found_new_row = None
        
        # Limit search time for this step
        step_start = time.time()
        while target_max >= lower_bound and time.time() - step_start < 0.5:
            new_row = _greedy_row_with_constraint(k, other_diffs, target_max, rng)
            if new_row is not None:
                # Check if this row's diffs conflict with each other? 
                # _greedy_row_with_constraint ensures distinctness within the row.
                # And it avoids other_diffs.
                found_new_row = new_row
                break
            target_max -= 1
        
        if found_new_row is not None:
            # Update solution
            best_solution[max_idx] = found_new_row
            new_scope = max(max(row) for row in best_solution)
            if new_scope < best_scope:
                best_scope = new_scope
                # Update mask? We can just recalculate if needed, or maintain set.
                # For speed, let's just recalculate mask if we use bitmask logic later.
                # Here we use sets, so we just update the solution.
                improved = True
                if best_scope <= lower_bound:
                    return best_solution

        # If no scope reduction, try local compaction of the current best solution
        if not improved:
            # Try to shift elements down in any row
            compaction_improved = False
            for r_idx in range(n_rows):
                row = best_solution[r_idx]
                # Get diffs from other rows
                other_diffs = set()
                for r2 in range(n_rows):
                    if r2 == r_idx:
                        continue
                    r = best_solution[r2]
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                # Try to compact this row
                new_row = _compact_row(row, other_diffs, k, rng)
                if new_row is not None and max(new_row) < max(row):
                    best_solution[r_idx] = new_row
                    compaction_improved = True
                    # Recalculate global scope
                    new_scope = max(max(row) for row in best_solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        if best_scope <= lower_bound:
                            return best_solution
            
            # If compaction didn't help, try random restarts of a single row
            if not compaction_improved and time.time() < deadline - 5.0:
                # Pick a random row and try to regenerate it with a random target scope
                r_idx = rng.randint(0, n_rows - 1)
                other_diffs = set()
                for r2 in range(n_rows):
                    if r2 == r_idx:
                        continue
                    r = best_solution[r2]
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                # Try to regenerate with a scope slightly smaller than current max of that row
                current_max = best_solution[r_idx][-1]
                target_max = current_max - rng.randint(1, 5)
                if target_max > lower_bound:
                    new_row = _greedy_row_with_constraint(k, other_diffs, target_max, rng)
                    if new_row is not None:
                        best_solution[r_idx] = new_row
                        new_scope = max(max(row) for row in best_solution)
                        if new_scope < best_scope:
                            best_scope = new_scope

    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _greedy_row_with_constraint(k, forbidden_diffs, max_val, rng):
    """
    Construct a row of k+1 integers starting at 0, ending <= max_val,
    such that all pairwise differences are distinct and not in forbidden_diffs.
    """
    row = [0]
    # We need to pick k more numbers.
    # To keep the max small, we should pick numbers that are "tight" but valid.
    # Greedy: pick the smallest possible next number.
    
    # However, pure greedy might fail. We can use a simple backtracking or randomization.
    # Given the small size (k=7), a simple randomized greedy is good.
    
    attempts = 10
    for _ in range(attempts):
        row = [0]
        success = True
        for _ in range(k):
            # Find candidates
            last = row[-1]
            # Next number must be > last and <= max_val
            # Also, all diffs with previous numbers must be valid
            candidates = []
            # Sample a range of candidates
            # To encourage small max, we look near the end? No, we want to fit in max_val.
            # Let's just try values from last+1 to max_val.
            # To speed up, we can randomize the order or take the first valid one.
            
            # Optimization: Check from max_val downwards? No, we want to leave room for future numbers.
            # Actually, if we pick small numbers early, we have more room.
            # So pick the smallest valid number.
            
            found = False
            # We need to ensure we can still place k - len(row) more numbers.
            # Max possible value for the last number is max_val.
            # Min increment is 1.
            
            for x in range(last + 1, max_val + 1):
                valid = True
                for a in row:
                    d = x - a
                    if d in forbidden_diffs:
                        valid = False
                        break
                    # Also check if d is already used in this row?
                    # We maintain a set of used diffs for this row.
                    # Wait, the `forbidden_diffs` passed in is only from OTHER rows.
                    # We must also ensure distinctness WITHIN this row.
                    # Let's maintain a local set.
                # Re-do logic with local set
                
            # Let's restart the loop with a local set
            local_used = set()
            row = [0]
            for _ in range(k):
                last = row[-1]
                found_val = None
                # Try values in a randomized order or sorted order?
                # Sorted order (smallest first) tends to produce smaller max.
                for x in range(last + 1, max_val + 1):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in forbidden_diffs or d in local_used:
                            valid = False
                            break
                    if valid:
                        # Check if we can complete the row
                        # We have k - len(row) slots left.
                        # Current max is x.
                        # We need to place remaining numbers > x, <= max_val.
                        # This is a simple check.
                        found_val = x
                        break
                
                if found_val is None:
                    success = False
                    break
                
                # Add diffs to local_used
                for a in row:
                    local_used.add(found_val - a)
                row.append(found_val)
            
            if success:
                return row
            
            # If failed, randomize next attempt?
            # The greedy "smallest first" is deterministic.
            # To allow different structures, we can skip some small candidates.
            # For simplicity, let's just return None if greedy fails.
            # But we can add a bit of randomness:
            # In the candidate loop, if we find a valid x, we accept it with prob 0.8, else skip?
            # Or just try again with a different random seed?
            # The function takes rng. Let's use it to shuffle candidate order slightly.
            # Actually, let's just try the greedy approach once. If it fails, return None.
            # The caller will try other target_max values.
            
    return None

def _compact_row(row, forbidden_diffs, k, rng):
    """
    Try to reduce the max entry of a single row by shifting elements down.
    """
    original_max = max(row)
    # Try to shift each element (from right to left) to the smallest valid position.
    new_row = list(row)
    local_used = set()
    
    # We need to be careful: shifting an element changes diffs with all other elements.
    # It's safer to rebuild the row greedily with a tighter bound.
    # Let's just use _greedy_row_with_constraint with max_val = original_max - 1.
    # If that works, we return it.
    
    # But we want to keep the "structure" somewhat similar?
    # No, any valid row is fine.
    
    # Let's try to construct a row with max < original_max.
    new_row = _greedy_row_with_constraint(k, forbidden_diffs, original_max - 1, rng)
    if new_row:
        return new_row
    return None
