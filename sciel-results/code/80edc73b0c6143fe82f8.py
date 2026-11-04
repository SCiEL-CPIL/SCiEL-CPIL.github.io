"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Greedy baseline with bitset optimization
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            if randomize:
                if len(row) > 2 and random.random() < 0.3:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows

    # Simulated Annealing
    def simulated_annealing(initial_rows, deadline_t, T0=10.0, cooling_rate=0.995):
        rows = [list(r) for r in initial_rows]
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        current_max = best_max
        
        # Precompute diff sets for current rows
        def compute_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs
        
        row_diffs = [compute_diffs(row) for row in rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        def is_valid_rows(rows_list, rd_list):
            """Check if rows are valid (increasing, start at 0, unique diffs)"""
            for i, row in enumerate(rows_list):
                if row[0] != 0:
                    return False
                for j in range(len(row)-1):
                    if row[j] >= row[j+1]:
                        return False
            # Check all diffs unique
            seen = set()
            for rd in rd_list:
                if rd & seen:
                    return False
                seen |= rd
            return True
        
        T = T0
        step = 0
        
        while time.time() < deadline_t:
            step += 1
            if step % 10000 == 0:
                T *= cooling_rate
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Swap elements between two different rows
                if n_rows < 2:
                    continue
                r1, r2 = random.sample(range(n_rows), 2)
                i1 = random.randint(1, len(rows[r1]) - 1)
                i2 = random.randint(1, len(rows[r2]) - 1)
                
                # Save old state
                old_r1 = list(rows[r1])
                old_r2 = list(rows[r2])
                old_rd1 = set(row_diffs[r1])
                old_rd2 = set(row_diffs[r2])
                old_all = set(all_diffs)
                
                # Perform swap
                rows[r1][i1], rows[r2][i2] = rows[r2][i2], rows[r1][i1]
                
                # Check if rows are still increasing
                if not all(rows[r1][j] < rows[r1][j+1] for j in range(len(rows[r1])-1)) or \
                   not all(rows[r2][j] < rows[r2][j+1] for j in range(len(rows[r2])-1)):
                    # Rollback
                    rows[r1] = old_r1
                    rows[r2] = old_r2
                    row_diffs[r1] = old_rd1
                    row_diffs[r2] = old_rd2
                    all_diffs = old_all
                    continue
                
                # Compute new diffs
                new_rd1 = compute_diffs(rows[r1])
                new_rd2 = compute_diffs(rows[r2])
                
                # Check for duplicate diffs within rows
                if len(new_rd1) != len(rows[r1])*(len(rows[r1])-1)//2 or \
                   len(new_rd2) != len(rows[r2])*(len(rows[r2])-1)//2:
                    rows[r1] = old_r1
                    rows[r2] = old_r2
                    row_diffs[r1] = old_rd1
                    row_diffs[r2] = old_rd2
                    all_diffs = old_all
                    continue
                
                # Check for conflicts
                new_all = (all_diffs - old_rd1 - old_rd2) | new_rd1 | new_rd2
                if len(new_all) != len(all_diffs - old_rd1 - old_rd2) + len(new_rd1) + len(new_rd2):
                    # Conflict: some diff appears in both new_rd1 and new_rd2, or in remaining
                    rows[r1] = old_r1
                    rows[r2] = old_r2
                    row_diffs[r1] = old_rd1
                    row_diffs[r2] = old_rd2
                    all_diffs = old_all
                    continue
                
                # Accept
                new_max = get_max_entry(rows)
                delta = new_max - current_max
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_max = new_max
                    row_diffs[r1] = new_rd1
                    row_diffs[r2] = new_rd2
                    all_diffs = new_all
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                else:
                    # Rollback
                    rows[r1] = old_r1
                    rows[r2] = old_r2
                    row_diffs[r1] = old_rd1
                    row_diffs[r2] = old_rd2
                    all_diffs = old_all
                    
            elif move_type < 0.6:
                # Replace an element in a row with a nearby value
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, len(rows[r]) - 1)
                
                # Try values near current
                current_val = rows[r][i]
                candidates = []
                for delta_val in range(-5, 6):
                    if delta_val == 0:
                        continue
                    new_val = current_val + delta_val
                    if new_val > 0:
                        candidates.append(new_val)
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                
                for new_val in candidates:
                    # Check if row would still be increasing
                    if i > 0 and new_val <= rows[r][i-1]:
                        continue
                    if i < len(rows[r]) - 1 and new_val >= rows[r][i+1]:
                        continue
                    
                    # Save old state
                    old_row = list(rows[r])
                    old_rd = set(row_diffs[r])
                    old_all = set(all_diffs)
                    
                    # Perform change
                    rows[r][i] = new_val
                    
                    # Compute new diffs
                    new_rd = compute_diffs(rows[r])
                    
                    # Check for duplicate diffs within row
                    if len(new_rd) != len(rows[r])*(len(rows[r])-1)//2:
                        rows[r] = old_row
                        row_diffs[r] = old_rd
                        all_diffs = old_all
                        continue
                    
                    # Check for conflicts with other rows
                    new_all = (all_diffs - old_rd) | new_rd
                    if len(new_all) != len(all_diffs - old_rd) + len(new_rd):
                        rows[r] = old_row
                        row_diffs[r] = old_rd
                        all_diffs = old_all
                        continue
                    
                    # Accept
                    new_max = get_max_entry(rows)
                    delta = new_max - current_max
                    
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        current_max = new_max
                        row_diffs[r] = new_rd
                        all_diffs = new_all
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in rows]
                        break
                    else:
                        # Rollback
                        rows[r] = old_row
                        row_diffs[r] = old_rd
                        all_diffs = old_all
            
            else:
                # Rebuild one row from scratch using greedy with randomization
                r = random.randint(0, n_rows - 1)
                
                # Save old state
                old_row = list(rows[r])
                old_rd = set(row_diffs[r])
                old_all = set(all_diffs)
                
                # Build new row
                other_diffs = all_diffs - old_rd
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    new_diffs = set()
                    conflict = False
                    for a in new_row:
                        d = x - a
                        if d in other_diffs or d in new_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        new_row.append(x)
                
                if len(new_row) != k + 1:
                    continue
                
                # Check validity
                if not all(new_row[j] < new_row[j+1] for j in range(len(new_row)-1)):
                    continue
                
                new_rd = set(new_diffs)
                new_all = other_diffs | new_rd
                
                # Accept
                rows[r] = new_row
                new_max = get_max_entry(rows)
                delta = new_max - current_max
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_max = new_max
                    row_diffs[r] = new_rd
                    all_diffs = new_all
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                else:
                    # Rollback
                    rows[r] = old_row
                    row_diffs[r] = old_rd
                    all_diffs = old_all
        
        return best_rows, best_max
    
    # Main search
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    search_deadline = deadline - 5.0
    
    # Run simulated annealing with multiple restarts
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > search_deadline:
            break
        
        remaining = search_deadline - time.time()
        time_slice = remaining / (num_restarts - restart)
        time_slice = min(time_slice, 30.0)
        time_slice = max(time_slice, 5.0)
        
        deadline_for_this = time.time() + time_slice
        
        # Use different initial solutions for each restart
        if restart == 0:
            initial = best_rows
        else:
            initial = greedy_solution(randomize=True, seed=restart * 1000 + 42)
        
        try:
            improved_rows, improved_max = simulated_annealing(initial, deadline_for_this)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Final validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    if not validate(best_rows):
        # Fallback to pure greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
