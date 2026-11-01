import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Keep a margin from the 150s limit
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # First, get a valid greedy solution as fallback
    used_diffs_greedy = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs_greedy:
                    break
                x += 1
            row.append(x)
            current_max = x
            used_diffs_greedy |= new_diffs
        greedy_rows.append(row)
    
    greedy_scope = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    best_scope = greedy_scope
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def count_conflicts(row, used_diffs):
        """Count how many differences in row conflict with used_diffs."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return len(diffs & used_diffs)
    
    def apply_diff_changes(row, used_diffs, old_val, new_val, idx):
        """
        Update used_diffs when row[idx] changes from old_val to new_val.
        Returns new used_diffs set.
        """
        new_used = set(used_diffs)
        # Remove old differences involving old_val
        for j in range(len(row)):
            if j != idx:
                if j < idx:
                    d = old_val - row[j]
                    new_used.discard(d)
                else:
                    d = row[j] - old_val
                    new_used.discard(d)
        # Add new differences involving new_val
        for j in range(len(row)):
            if j != idx:
                if j < idx:
                    d = new_val - row[j]
                    new_used.add(d)
                else:
                    d = row[j] - new_val
                    new_used.add(d)
        return new_used
    
    def backtrack_row(row, used_diffs, depth, max_val_bound):
        """
        Try to complete a row starting from given state.
        row is a list, used_diffs is the set of all used differences.
        Returns (completed_row, new_used_diffs) or None if failed.
        """
        if time.time() > deadline - 5.0:
            return None
        
        if len(row) == k + 1:
            return (row[:], used_diffs)
        
        prev = row[-1]
        
        # Try candidate values
        # Start from prev+1, with some randomization for diversity
        base_candidates = []
        upper = max(prev + 30, prev + (k + 1 - len(row)) + 10)
        upper = min(upper, max_val_bound + 5)
        
        for x in range(prev + 1, upper + 1):
            if time.time() > deadline - 5.0:
                return None
            
            # Check if adding x creates any conflict with used_diffs
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            
            # Forward checking: ensure we can still complete the row
            # Simple heuristic: check that there's enough "space"
            
            new_row = row + [x]
            new_used = used_diffs | new_diffs
            
            # If we're at the last position, we're done
            if len(new_row) == k + 1:
                return (new_row, new_used)
            
            # Recurse
            result = backtrack_row(new_row, new_used, depth + 1, max_val_bound)
            if result is not None:
                return result
        
        return None
    
    # Strategy: Use simulated annealing on the binding (largest) row first
    # then polish other rows
    
    def sa_optimize(rows, used_diffs, num_iterations=5000):
        """
        Simulated annealing to optimize the rows.
        Focus on reducing the scope (largest entry).
        """
        current_rows = [r[:] for r in rows]
        current_used = set(used_diffs)
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 10.0  # Initial temperature
        cooling = 0.999
        
        for it in range(num_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            
            # Pick a random position to change (not the first element 0)
            if len(row) <= 1:
                continue
            pos = random.randint(1, len(row) - 1)
            
            old_val = row[pos]
            
            # Generate a new candidate value
            # Try to move towards reducing the scope
            if row[pos] == max(current_rows[row_idx]) and current_scope == max(max(r) for r in current_rows):
                # This is the binding element, try to reduce it
                delta = random.randint(-5, 2)
                new_val = old_val + delta
            else:
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            
            # Ensure strictly increasing
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Check if the new value creates conflicts
            # Compute differences involving the changed position
            old_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        old_diffs.add(old_val - row[j])
                    else:
                        old_diffs.add(row[j] - old_val)
            
            new_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        new_diffs.add(new_val - row[j])
                    else:
                        new_diffs.add(row[j] - new_val)
            
            # Check conflicts with other rows' differences
            # We need to ensure new_diffs don't conflict with differences in other rows
            # that are not in old_diffs
            other_diffs = set()
            for i, r in enumerate(current_rows):
                if i == row_idx:
                    continue
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        other_diffs.add(r[b] - r[a])
            
            # Also need to check within the same row for other differences
            same_row_other_diffs = set()
            for a in range(len(row)):
                for b in range(a+1, len(row)):
                    if a != pos and b != pos:
                        same_row_other_diffs.add(row[b] - row[a])
            
            # New differences must not conflict with:
            # 1. Differences in other rows
            # 2. Differences in the same row that don't involve the changed position
            # 3. But old differences involving the changed position are being replaced
            
            conflict = (new_diffs & other_diffs) or (new_diffs & same_row_other_diffs)
            
            if conflict:
                continue
            
            # Compute the change in scope
            old_row_max = max(row)
            new_row = row[:]
            new_row[pos] = new_val
            new_row_max = max(new_row)
            
            old_scope = current_scope
            new_scope = max(max(r) for i, r in enumerate(current_rows) if i != row_idx) 
            new_scope = max(new_scope, new_row_max)
            
            delta_scope = new_scope - old_scope
            
            # Accept if better, or with probability based on temperature
            if delta_scope <= 0 or random.random() < math.exp(-delta_scope / T):
                # Accept the move
                # Update used_diffs
                # Remove old differences involving old_val
                for j in range(len(row)):
                    if j != pos:
                        if j < pos:
                            d = old_val - row[j]
                            current_used.discard(d)
                        else:
                            d = row[j] - old_val
                            current_used.discard(d)
                # Add new differences involving new_val
                for j in range(len(row)):
                    if j != pos:
                        if j < pos:
                            d = new_val - row[j]
                            current_used.add(d)
                        else:
                            d = row[j] - new_val
                            current_used.add(d)
                
                current_rows[row_idx] = new_row
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            
            T *= cooling
        
        return best_rows, best_scope
    
    # Try multiple random restarts with different initial configurations
    num_restarts = 20
    best_found = best_scope
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Start with the greedy solution as base
        used_diffs = set()
        current_rows = []
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline - 5.0:
                success = False
                break
            
            # Try to construct this row with backtracking
            row = [0]
            
            # Use backtracking to complete the row
            result = backtrack_row(row, used_diffs, 0, best_found + 10)
            
            if result is None:
                # Backtracking failed for this row, use greedy
                row = [0]
                current_max = 0
                while len(row) < k + 1:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            break
                        x += 1
                    row.append(x)
                    current_max = x
                    used_diffs |= new_diffs
            else:
                completed_row, used_diffs = result
                row = completed_row
            
            current_rows.append(row)
        
        if not success:
            continue
        
        # Now optimize with simulated annealing
        if time.time() > deadline - 10.0:
            continue
        
        optimized_rows, opt_scope = sa_optimize(current_rows, used_diffs, num_iterations=3000)
        
        if opt_scope < best_found:
            best_found = opt_scope
            best_rows = [r[:] for r in optimized_rows]
            best_scope = opt_scope
    
    # If we didn't improve, return the greedy solution
    if best_found >= greedy_scope:
        return greedy_rows
    
    return best_rows
