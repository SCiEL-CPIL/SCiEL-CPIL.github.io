import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Simulated Annealing with efficient conflict tracking.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Greedy baseline
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
    
    # Initial solution
    rows = greedy_construct(n_rows, k)
    current_max = max(max(r) for r in rows)
    
    # State variables for SA
    # diff_count[d] = number of times difference d appears
    # We need to know the max possible difference to size the array.
    # Greedy max is a good upper bound. Let's allow some room.
    max_possible_diff = current_max + 50 
    diff_count = [0] * (max_possible_diff + 1)
    num_conflicts = 0
    
    def update_diff(row_idx, old_val, new_val, row):
        """
        Updates diff_count and num_conflicts when row[row_idx] changes from old_val to new_val.
        Returns the change in num_conflicts (delta_conflicts) and whether new_val caused index out of bounds (handled outside).
        """
        nonlocal num_conflicts
        
        # Remove old diffs
        # Diffs involving old_val:
        # 1. old_val - row[j] for j < row_idx
        # 2. row[j] - old_val for j > row_idx
        
        # We need to be careful: row is currently in the state with old_val.
        # So row[row_idx] is old_val.
        
        for j in range(row_idx):
            d = old_val - row[j]
            if d > 0:
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    num_conflicts -= 1
                # if it was 0, it's -1 (error), but we assume state is consistent
                
        for j in range(row_idx + 1, len(row)):
            d = row[j] - old_val
            if d > 0:
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    num_conflicts -= 1

        # Set new value in row temporarily for calculation? 
        # The caller will update row[row_idx] = new_val.
        # But we need to calculate new diffs using new_val and other elements of row.
        # The other elements are unchanged.
        
        # Add new diffs
        for j in range(row_idx):
            d = new_val - row[j]
            if d > 0:
                if d >= len(diff_count):
                    # Expand array if necessary (should be rare if max_possible_diff is large enough)
                    # But we check bounds before calling this.
                    pass 
                diff_count[d] += 1
                if diff_count[d] == 2:
                    num_conflicts += 1

        for j in range(row_idx + 1, len(row)):
            d = row[j] - new_val
            if d > 0:
                if d >= len(diff_count):
                    pass
                diff_count[d] += 1
                if diff_count[d] == 2:
                    num_conflicts += 1

    # Initialize diff_count with the greedy solution
    for r in range(n_rows):
        row = rows[r]
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d < len(diff_count):
                    diff_count[d] += 1
                    if diff_count[d] == 2:
                        num_conflicts += 1
                else:
                    # Expand if needed
                    diff_count.extend([0] * (d - len(diff_count) + 1))
                    diff_count[d] += 1
                    if diff_count[d] == 2:
                        num_conflicts += 1

    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s safety margin for 120s limit
    
    T = 100.0 # Initial temperature
    cooling_rate = 0.999
    
    # We want to minimize Cost = 10000 * num_conflicts + max_entry
    # Current cost
    current_cost = 10000 * num_conflicts + current_max
    
    # Precompute row lengths
    row_len = k + 1
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        # Pick an index to change. Index 0 is fixed at 0.
        # We can change indices 1 to k.
        i_idx = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[i_idx]
        
        # Determine valid range for new_val
        # Must be > row[i_idx - 1]
        # Must be < row[i_idx + 1] if i_idx < k
        lower = row[i_idx - 1] + 1
        upper = row[i_idx + 1] - 1 if i_idx < k else max_possible_diff + 50
        
        if lower > upper:
            T *= cooling_rate
            continue
            
        # Choose a new value
        # Bias towards smaller values to minimize max_entry, but allow exploration
        # Uniform random in [lower, upper]
        # Or, pick a value near old_val?
        # Let's try a mix: 50% local (near old_val), 50% global (random in range)
        if random.random() < 0.5:
            # Local move
            span = max(1, (upper - lower) // 10)
            new_val = random.randint(max(lower, old_val - span), min(upper, old_val + span))
        else:
            # Global move within valid range
            new_val = random.randint(lower, upper)
            
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        # Check if new_val exceeds our diff_count array size
        # Max diff involving new_val is max(new_val, max(row))
        # If new_val is large, diffs can be large.
        # We allocated max_possible_diff + 50.
        # If new_val > max_possible_diff, we might need to expand.
        # But we want to minimize max, so large values are bad.
        # If new_val is huge, cost will be huge, so it will likely be rejected.
        # But we must not crash.
        
        if new_val >= len(diff_count):
            # Expand array
            diff_count.extend([0] * (new_val - len(diff_count) + 10))
            
        # Calculate delta cost
        # We need to simulate the change in num_conflicts.
        # Instead of updating global state and reverting, let's calculate the change.
        
        # This is tricky because num_conflicts depends on the entire array.
        # It's easier to update, check, and revert if rejected.
        
        # Save state
        # We need to revert diff_count changes and num_conflicts.
        # We can store the diffs that were changed.
        
        changed_diffs = []
        
        # Remove old
        for j in range(i_idx):
            d = old_val - row[j]
            if d > 0:
                changed_diffs.append((d, -1))
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    num_conflicts -= 1
                    
        for j in range(i_idx + 1, len(row)):
            d = row[j] - old_val
            if d > 0:
                changed_diffs.append((d, -1))
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    num_conflicts -= 1
                    
        # Add new
        for j in range(i_idx):
            d = new_val - row[j]
            if d > 0:
                changed_diffs.append((d, 1))
                diff_count[d] += 1
                if diff_count[d] == 2:
                    num_conflicts += 1
                    
        for j in range(i_idx + 1, len(row)):
            d = row[j] - new_val
            if d > 0:
                changed_diffs.append((d, 1))
                diff_count[d] += 1
                if diff_count[d] == 2:
                    num_conflicts += 1
                    
        # Update row
        row[i_idx] = new_val
        
        # Calculate new max
        # We can just compute max(row) for this row and compare with others?
        # No, global max.
        # Since n_rows is small, max(max(r) for r in rows) is fast.
        new_max = max(max(r) for r in rows)
        
        new_cost = 10000 * num_conflicts + new_max
        
        delta_cost = new_cost - current_cost
        
        # Acceptance probability
        if delta_cost < 0:
            accept = True
        else:
            if T > 1e-6:
                accept = random.random() < math.exp(-delta_cost / T)
            else:
                accept = False
                
        if accept:
            current_cost = new_cost
            current_max = new_max
            if num_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            row[i_idx] = old_val
            
            # Revert diff_count and num_conflicts
            # We stored the changes. We need to reverse them.
            # The list `changed_diffs` contains (d, delta) where delta is +1 for add, -1 for remove.
            # To revert, we apply -delta.
            for d, delta in changed_diffs:
                # Reverse the change
                reverse_delta = -delta
                old_count = diff_count[d]
                diff_count[d] += reverse_delta
                new_count = diff_count[d]
                
                # Update num_conflicts based on transition
                # Transition: old_count -> new_count
                # Conflict count changes if:
                # 1->0 (no change in conflicts? No, 1 is not conflict, 0 is not. Wait.)
                # Conflict is count > 1.
                # If old_count was 2 and new_count is 1: conflict removed (-1)
                # If old_count was 1 and new_count is 2: conflict added (+1)
                # If old_count was 3 and new_count is 2: no change (both > 1)
                # If old_count was 2 and new_count is 3: no change
                # If old_count was 1 and new_count is 0: no change
                # If old_count was 0 and new_count is 1: no change
                
                if old_count == 2 and new_count == 1:
                    num_conflicts -= 1
                elif old_count == 1 and new_count == 2:
                    num_conflicts += 1
                # Other transitions don't affect num_conflicts
                
            # Verify num_conflicts is consistent? 
            # It's hard to verify quickly. Trust the logic.
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Keep a minimum temperature to allow some exploration
            
    # Return best valid solution found
    # If best_rows has conflicts, we should return the greedy one?
    # The problem asks for a valid set.
    # If SA didn't find a valid one better than greedy, return greedy.
    # But greedy is valid.
    # best_rows is updated only if num_conflicts == 0.
    # So best_rows is always valid.
    
    return best_rows
