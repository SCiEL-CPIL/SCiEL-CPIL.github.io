import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n_marks = k + 1  # marks per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Greedy initialization
    def greedy_init():
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Fast cost computation using precomputed difference sets
    def compute_cost_fast(rows):
        """Compute cost: max_entry + conflicts * 100"""
        used = set()
        max_entry = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            max_entry = max(max_entry, row[-1])
            for i in range(n_marks):
                for j in range(i + 1, n_marks):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_entry + conflicts * 100
    
    # Incremental cost: compute only the delta when changing one position
    def get_row_diffs(row):
        """Get all positive differences in a row"""
        diffs = set()
        for i in range(n_marks):
            for j in range(i + 1, n_marks):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Simulated Annealing with better moves
    def simulated_annealing(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost_fast(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        # Precompute difference sets for each row for fast incremental updates
        row_diffs = [get_row_diffs(r) for r in current_rows]
        
        T_start = 80.0
        T_end = 0.01
        steps = 0
        max_steps = 300000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))
        
        # Upper bound for random values
        val_bound = max(r[-1] for r in current_rows) + 50
        
        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * (cooling_rate ** steps)
            if T < T_end:
                T = T_end
            
            # Random move: pick a random row and a random mark position (1..k)
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = current_rows[r][pos]
            
            # Determine valid range
            min_val = current_rows[r][pos - 1] + 1
            max_val = current_rows[r][pos + 1] - 1 if pos < k else val_bound
            
            if min_val > max_val:
                continue
            
            # Choose new value - bias towards smaller values to reduce scope
            if random.random() < 0.3:
                # Try to reduce: pick from lower half of range
                new_val = random.randint(min_val, min_val + max(1, (max_val - min_val) // 2))
            elif random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            else:
                # Random in range
                new_val = random.randint(min_val, max_val)
            
            if new_val < min_val or new_val > max_val:
                continue
            
            if new_val == old_val:
                continue
            
            # Compute incremental cost change
            # Old diffs that will be removed
            old_row = current_rows[r]
            old_diffs = row_diffs[r]
            
            # New row with updated value
            new_row = old_row[:]
            new_row[pos] = new_val
            new_diffs = get_row_diffs(new_row)
            
            # Diffs being removed from this row
            removed = old_diffs - new_diffs
            # Diffs being added to this row
            added = new_diffs - old_diffs
            
            # Check if any added diff conflicts with other rows
            # We need to know which diffs are used by other rows
            # For speed, we'll just compute full cost if the move looks promising
            # Quick check: if added diffs don't conflict with any other row's diffs
            conflict_count = 0
            for d in added:
                if d <= 0:
                    conflict_count += 100
                    continue
                # Check against all other rows
                for r2 in range(n_rows):
                    if r2 == r:
                        continue
                    if d in row_diffs[r2]:
                        conflict_count += 1
                        break
            
            # Estimate delta cost
            # max_entry change
            old_max = max_entry if 'max_entry' in dir() else current_rows[r][-1]
            # Recompute max for this row
            new_row_max = max(new_row)
            old_row_max = max(old_row)
            
            # Simple: just compute full cost (it's fast enough for k=8, n_rows=4)
            current_rows[r] = new_row
            new_cost = compute_cost_fast(current_rows)
            row_diffs[r] = new_diffs
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(row) for row in current_rows]
            else:
                # Revert
                current_rows[r] = old_row
                row_diffs[r] = old_diffs
            
        return best_rows, best_cost
    
    # Local search: try to fix conflicts by adjusting individual marks
    def local_search(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost_fast(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        no_improve = 0
        max_no_improve = 3000
        val_bound = max(r[-1] for r in current_rows) + 30
        
        while time.time() < time_budget and no_improve < max_no_improve:
            improved = False
            # Try each row and each position
            for r in range(n_rows):
                for pos in range(1, k + 1):
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else val_bound
                    
                    if min_val > max_val:
                        continue
                    
                    best_local_val = current_rows[r][pos]
                    best_local_cost = current_cost
                    
                    # Try a few candidate values - focus on reducing scope
                    candidates = set()
                    candidates.add(current_rows[r][pos])
                    candidates.add(min_val)
                    if current_rows[r][pos] > min_val:
                        candidates.add(current_rows[r][pos] - 1)
                        candidates.add(current_rows[r][pos] - 2)
                    if current_rows[r][pos] < max_val:
                        candidates.add(current_rows[r][pos] + 1)
                    # Add some random candidates biased towards smaller values
                    for _ in range(4):
                        candidates.add(random.randint(min_val, min_val + max(1, (max_val - min_val) // 3)))
                    
                    for val in candidates:
                        if val < min_val or val > max_val:
                            continue
                        current_rows[r][pos] = val
                        new_cost = compute_cost_fast(current_rows)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = val
                    
                    current_rows[r][pos] = best_local_val
                    if best_local_cost < current_cost:
                        current_cost = best_local_cost
                        improved = True
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(row) for row in current_rows]
        
        return best_rows, best_cost
    
    # Final polish: try to reduce the maximum entry by shifting
    def final_polish(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost_fast(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        # Try to reduce the maximum entry
        for _ in range(50):
            if time.time() > time_budget:
                break
            
            # Find the row with the largest entry
            max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
            max_val = current_rows[max_r][-1]
            
            if max_val <= 0:
                break
            
            # Try to shift the entire row down by 1
            shifted = [0] + [v - 1 for v in current_rows[max_r][1:]]
            if all(shifted[i] < shifted[i + 1] for i in range(n_marks - 1)):
                old_row = current_rows[max_r]
                current_rows[max_r] = shifted
                temp_cost = compute_cost_fast(current_rows)
                if temp_cost <= current_cost:
                    current_cost = temp_cost
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_rows = [list(row) for row in current_rows]
                else:
                    current_rows[max_r] = old_row
            else:
                break
        
        return best_rows, best_cost
    
    # Get initial solution
    greedy_rows = greedy_init()
    
    # Run simulated annealing
    sa_rows, sa_cost = simulated_annealing(greedy_rows, time.time() + 35)
    
    # Run local search from the SA result
    ls_rows, ls_cost = local_search(sa_rows, time.time() + 70)
    
    # Final polish
    polished_rows, polished_cost = final_polish(ls_rows, time.time() + 85)
    
    best_rows = polished_rows
    best_cost = polished_cost
    
    # Try a few random restarts if time permits
    remaining = deadline - time.time()
    if remaining > 15:
        for restart in range(4):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = [list(r) for r in best_rows]
            for _ in range(8):
                r = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                min_val = perturbed[r][pos - 1] + 1
                max_val = perturbed[r][pos + 1] - 1 if pos < k else max(perturbed[r][-1] + 20, 50)
                if min_val <= max_val:
                    perturbed[r][pos] = random.randint(min_val, max_val)
            
            restart_budget = min(deadline - 5, time.time() + 12)
            sa2_rows, sa2_cost = simulated_annealing(perturbed, restart_budget)
            ls2_rows, ls2_cost = local_search(sa2_rows, min(deadline - 5, time.time() + 10))
            if ls2_cost < best_cost:
                best_cost = ls2_cost
                best_rows = ls2_rows
    
    # Final validation and cleanup
    # Ensure strictly increasing and non-negative
    for r in range(n_rows):
        for m in range(1, n_marks):
            if best_rows[r][m] <= best_rows[r][m - 1]:
                best_rows[r][m] = best_rows[r][m - 1] + 1
        # Ensure all values are non-negative
        for m in range(n_marks):
            if best_rows[r][m] < 0:
                best_rows[r][m] = 0
            if m > 0 and best_rows[r][m] <= best_rows[r][m - 1]:
                best_rows[r][m] = best_rows[r][m - 1] + 1
    
    return best_rows
