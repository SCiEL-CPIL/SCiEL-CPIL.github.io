"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Compute all differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Check if a row is valid (strictly increasing, starts at 0)
    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False
        return True
    
    # Check if a set of rows is valid (all differences distinct across all rows)
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if not is_valid_row(row):
                return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True
    
    # Greedy construction
    def greedy_solution(seed=None):
        if seed is not None:
            random.seed(seed)
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
    
    # Simulated annealing
    def simulated_annealing(initial_rows, deadline_t, max_iter=5000):
        rows = [list(r) for r in initial_rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= row_diffs(row)
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        # Temperature schedule
        T0 = 10.0
        T_end = 0.1
        cooling = (T_end / T0) ** (1.0 / max_iter)
        T = T0
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(rows[row_idx])
            old_diffs = row_diffs(old_row)
            
            # Generate a new row by perturbing the old one
            new_row = list(old_row)
            
            # Choose a perturbation strategy
            strategy = random.random()
            if strategy < 0.5:
                # Modify one element
                if len(new_row) > 1:
                    pos = random.randint(1, len(new_row) - 1)
                    # Try to find a new value that maintains validity
                    old_val = new_row[pos]
                    # Try nearby values
                    for delta in [1, -1, 2, -2, 3, -3, 5, -5]:
                        new_val = old_val + delta
                        if new_val <= new_row[pos-1] or (pos < len(new_row)-1 and new_val >= new_row[pos+1]):
                            continue
                        # Check if this creates internal conflicts
                        temp_row = list(new_row)
                        temp_row[pos] = new_val
                        temp_diffs = row_diffs(temp_row)
                        if len(temp_diffs) != diffs_per_row:
                            continue
                        # Check conflicts with other rows
                        other_diffs = used_diffs - old_diffs
                        if temp_diffs & other_diffs:
                            continue
                        new_row = temp_row
                        break
            elif strategy < 0.8:
                # Swap two elements
                if len(new_row) > 2:
                    i1, i2 = random.sample(range(1, len(new_row)), 2)
                    new_row[i1], new_row[i2] = new_row[i2], new_row[i1]
                    # Re-sort but keep 0 at front
                    new_row = [0] + sorted(new_row[1:])
            else:
                # Random restart of this row within a range
                max_entry = max(new_row[-1], 10)
                new_row = [0]
                for pos in range(1, k + 1):
                    # Try random values
                    found = False
                    for _ in range(20):
                        v = random.randint(new_row[-1] + 1, max_entry)
                        new_diffs = {v - a for a in new_row}
                        if len(new_diffs) != len(new_row):
                            continue
                        other_diffs = used_diffs - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        new_row.append(v)
                        found = True
                        break
                    if not found:
                        break
                if len(new_row) != k + 1:
                    new_row = old_row  # Keep old row if failed
            
            # Check if new row is valid
            if not is_valid_row(new_row):
                T *= cooling
                continue
            
            new_diffs = row_diffs(new_row)
            if len(new_diffs) != diffs_per_row:
                T *= cooling
                continue
            
            # Check conflicts with other rows
            other_diffs = used_diffs - old_diffs
            if new_diffs & other_diffs:
                T *= cooling
                continue
            
            # Calculate cost difference
            old_max = get_max_entry(rows)
            new_rows = [list(r) for r in rows]
            new_rows[row_idx] = new_row
            new_max = get_max_entry(new_rows)
            
            delta_cost = new_max - old_max
            
            # Accept if improvement, or with probability based on temperature
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Update state
                used_diffs = other_diffs | new_diffs
                rows[row_idx] = new_row
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
            
            T *= cooling
        
        return best_rows, best_max
    
    # Multi-start with simulated annealing
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution
    greedy = greedy_solution()
    best_rows = greedy
    best_max = get_max_entry(best_rows)
    
    # Try multiple random seeds for greedy
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        try:
            g = greedy_solution(seed=seed)
            g_max = get_max_entry(g)
            if g_max < best_max:
                best_max = g_max
                best_rows = g
        except Exception:
            pass
    
    # Run simulated annealing on the best solution
    if time.time() < deadline - 5:
        try:
            improved_rows, improved_max = simulated_annealing(best_rows, deadline - 5, max_iter=3000)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # If time allows, try more annealing with different initializations
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            try:
                seed = restart * 1000 + 7
                g = greedy_solution(seed=seed)
                g_max = get_max_entry(g)
                
                improved_rows, improved_max = simulated_annealing(g, deadline - 15, max_iter=1000)
                if improved_max < best_max:
                    best_max = improved_max
                    best_rows = improved_rows
            except Exception:
                pass
    
    # Final validation
    if not validate(best_rows):
        # Fallback to pure greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
