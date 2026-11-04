"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
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
    
    # Backtracking with constraint propagation
    def backtrack_solve(max_val, time_limit=30.0):
        """Try to find a valid solution with all entries <= max_val using backtracking."""
        deadline_bt = time.time() + time_limit
        used_diffs = set()
        rows = [None] * n_rows
        
        def can_place(row, pos, val, row_diffs_so_far):
            """Check if placing val at position pos in row is valid."""
            if pos > 0 and val <= row[pos - 1]:
                return False
            if pos < len(row) - 1 and val >= row[pos + 1]:
                return False
            # Check new diffs
            new_diffs = set()
            for i in range(len(row)):
                if i == pos:
                    continue
                if row[i] is not None:
                    d = abs(val - row[i])
                    if d == 0:
                        return False
                    if d in used_diffs or d in row_diffs_so_far or d in new_diffs:
                        return False
                    new_diffs.add(d)
            return True, new_diffs
        
        def solve_row(row_idx, row, pos, row_diffs):
            """Recursively fill in a row."""
            if time.time() > deadline_bt:
                return False
            
            if pos == len(row):
                return True
            
            # Try values from smallest possible
            min_val = row[pos - 1] + 1 if pos > 0 else 1
            max_possible = max_val
            
            # Use heuristic: try values that create "popular" differences first
            candidates = list(range(min_val, max_possible + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                if val > max_val:
                    break
                
                # Check if val can be placed
                conflict = False
                new_diffs = set()
                for i in range(len(row)):
                    if row[i] is not None:
                        d = abs(val - row[i])
                        if d == 0 or d in used_diffs or d in row_diffs or d in new_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Place val
                row[pos] = val
                old_used = set(used_diffs)
                old_row_diffs = set(row_diffs)
                used_diffs.update(new_diffs)
                row_diffs.update(new_diffs)
                
                if solve_row(row_idx, row, pos + 1, row_diffs):
                    return True
                
                # Backtrack
                row[pos] = None
                used_diffs.clear()
                used_diffs.update(old_used)
                row_diffs.clear()
                row_diffs.update(old_row_diffs)
            
            return False
        
        # Try to solve all rows
        for ri in range(n_rows):
            if time.time() > deadline_bt:
                return None
            rows[ri] = [None] * (k + 1)
            rows[ri][0] = 0
            row_diffs = set()
            
            if not solve_row(ri, rows[ri], 1, row_diffs):
                return None
        
        return rows
    
    # Greedy with randomization
    def greedy_solution(seed=None, max_val=None):
        if seed is not None:
            random.seed(seed)
        if max_val is None:
            max_val = 200
        
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_val:
                    break
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used_diffs.update(new_diffs)
            
            if len(row) != k + 1:
                return None
            rows.append(row)
        
        return rows
    
    # Simulated Annealing with multiple move types
    def simulated_annealing(initial_rows, deadline_t, T0=50.0, cooling_rate=0.999):
        rows = [list(r) for r in initial_rows]
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        current_max = best_max
        
        row_diffs = [compute_row_diffs(row) for row in rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        T = T0
        step = 0
        no_improve_steps = 0
        
        while time.time() < deadline_t:
            step += 1
            if step % 5000 == 0:
                T *= cooling_rate
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Change a single element value
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, len(rows[r]) - 1)
                
                current_val = rows[r][i]
                # Try nearby values
                search_range = max(10, current_max // 10)
                candidates = []
                for dv in range(-search_range, search_range + 1):
                    if dv == 0:
                        continue
                    new_val = current_val + dv
                    if new_val > 0:
                        candidates.append(new_val)
                
                random.shuffle(candidates)
                
                for new_val in candidates[:20]:  # Limit candidates
                    # Check ordering
                    if i > 0 and new_val <= rows[r][i - 1]:
                        continue
                    if i < len(rows[r]) - 1 and new_val >= rows[r][i + 1]:
                        continue
                    
                    # Compute new diffs for this row
                    new_row = list(rows[r])
                    new_row[i] = new_val
                    new_rd = compute_row_diffs(new_row)
                    
                    # Check for internal conflicts
                    if len(new_rd) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    
                    # Check conflicts with other rows
                    other_diffs = all_diffs - row_diffs[r]
                    if new_rd & other_diffs:
                        continue
                    
                    # Accept with SA criterion
                    new_max = max(new_row[-1], max(max(row) for ri, row in enumerate(rows) if ri != r))
                    delta = new_max - current_max
                    
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        # Apply move
                        rows[r] = new_row
                        row_diffs[r] = new_rd
                        all_diffs = other_diffs | new_rd
                        current_max = new_max
                        
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(row) for row in rows]
                            no_improve_steps = 0
                        break
                    else:
                        continue
                else:
                    no_improve_steps += 1
            
            elif move_type < 0.7:
                # Swap two elements within the same row
                r = random.randint(0, n_rows - 1)
                if len(rows[r]) < 3:
                    continue
                i1, i2 = random.sample(range(1, len(rows[r])), 2)
                
                # Ensure swap maintains ordering
                if i1 > i2:
                    i1, i2 = i2, i1
                
                # After swap, check if ordering is maintained
                # rows[r][i1] and rows[r][i2] are swapped
                # Need: ... < rows[r][i1] < rows[r][i2] < ...
                # After swap: ... < rows[r][i2] < rows[r][i1] < ...
                # This only works if rows[r][i2] < rows[r][i1] and neighbors are ok
                
                v1 = rows[r][i1]
                v2 = rows[r][i2]
                
                # Check if swap maintains increasing order
                valid = True
                if i1 > 0 and v2 <= rows[r][i1 - 1]:
                    valid = False
                if i2 < len(rows[r]) - 1 and v1 >= rows[r][i2 + 1]:
                    valid = False
                if i1 + 1 == i2:
                    # Adjacent swap: need v2 < v1 for it to make sense, but they're already in order
                    # Actually swapping adjacent in an increasing sequence breaks it unless equal
                    valid = False
                else:
                    # Check middle elements
                    for m in range(i1 + 1, i2):
                        if not (v2 < rows[r][m] < v1):
                            valid = False
                            break
                
                if not valid:
                    no_improve_steps += 1
                    continue
                
                # Perform swap
                old_row = list(rows[r])
                old_rd = set(row_diffs[r])
                old_all = set(all_diffs)
                
                rows[r][i1], rows[r][i2] = v2, v1
                
                new_rd = compute_row_diffs(rows[r])
                if len(new_rd) != len(rows[r]) * (len(rows[r]) - 1) // 2:
                    rows[r] = old_row
                    no_improve_steps += 1
                    continue
                
                other_diffs = all_diffs - old_rd
                if new_rd & other_diffs:
                    rows[r] = old_row
                    no_improve_steps += 1
                    continue
                
                new_max = get_max_entry(rows)
                delta = new_max - current_max
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    row_diffs[r] = new_rd
                    all_diffs = other_diffs | new_rd
                    current_max = new_max
                    
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(row) for row in rows]
                        no_improve_steps = 0
                else:
                    rows[r] = old_row
                    no_improve_steps += 1
            
            else:
                # Rebuild one row using greedy with randomization
                r = random.randint(0, n_rows - 1)
                
                old_row = list(rows[r])
                old_rd = set(row_diffs[r])
                old_all = set(all_diffs)
                
                other_diffs = all_diffs - old_rd
                
                # Build new row with some randomization
                new_row = [0]
                x = 0
                max_x = current_max + 10
                attempts = 0
                while len(new_row) < k + 1 and x < max_x:
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
                    no_improve_steps += 1
                    continue
                
                new_rd = set(compute_row_diffs(new_row))
                if new_rd & other_diffs:
                    no_improve_steps += 1
                    continue
                
                new_max = max(new_row[-1], max(max(row) for ri, row in enumerate(rows) if ri != r))
                delta = new_max - current_max
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    rows[r] = new_row
                    row_diffs[r] = new_rd
                    all_diffs = other_diffs | new_rd
                    current_max = new_max
                    
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(row) for row in rows]
                        no_improve_steps = 0
                else:
                    no_improve_steps += 1
            
            # Restart if stuck
            if no_improve_steps > 10000:
                # Reset to best and perturb
                rows = [list(r) for r in best_rows]
                row_diffs = [compute_row_diffs(row) for row in rows]
                all_diffs = set()
                for rd in row_diffs:
                    all_diffs |= rd
                current_max = best_max
                no_improve_steps = 0
                T = max(T, 10.0)
        
        return best_rows, best_max
    
    # Main search
    # First, try backtracking with increasing max_val
    best_rows = None
    best_max = float('inf')
    
    # Try backtracking with a reasonable max_val
    bt_time_limit = min(20.0, deadline - start_time - 5.0)
    if bt_time_limit > 2.0:
        for mv in [120, 130, 140, 150]:
            if time.time() > start_time + bt_time_limit:
                break
            result = backtrack_solve(mv, time_limit=min(5.0, bt_time_limit - (time.time() - start_time)))
            if result is not None:
                best_rows = result
                best_max = get_max_entry(best_rows)
                break
    
    # If backtracking failed, use greedy
    if best_rows is None:
        for seed in [None, 42, 123, 456]:
            rows = greedy_solution(seed=seed)
            if rows is not None:
                if best_rows is None or get_max_entry(rows) < best_max:
                    best_rows = rows
                    best_max = get_max_entry(rows)
                if best_max <= 130:
                    break
    
    if best_rows is None:
        # Last resort: pure greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    # Now apply simulated annealing to improve
    search_deadline = deadline - 3.0
    
    # Multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > search_deadline:
            break
        
        remaining = search_deadline - time.time()
        time_slice = remaining / (num_restarts - restart)
        time_slice = min(time_slice, 25.0)
        time_slice = max(time_slice, 5.0)
        
        deadline_for_this = time.time() + time_slice
        
        # Use different initial solutions
        if restart == 0:
            initial = best_rows
        elif restart == 1:
            initial = greedy_solution(seed=restart * 7919 + 1)
            if initial is None:
                initial = best_rows
        elif restart == 2:
            initial = greedy_solution(seed=restart * 104729 + 7)
            if initial is None:
                initial = best_rows
        else:
            # Perturb best solution
            initial = [list(r) for r in best_rows]
            # Randomly perturb some elements
            for _ in range(3):
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, len(initial[r]) - 1)
                delta_val = random.randint(-5, 5)
                new_val = initial[r][i] + delta_val
                if new_val > 0:
                    # Check if valid
                    if i > 0 and new_val > initial[r][i - 1]:
                        if i < len(initial[r]) - 1 and new_val < initial[r][i + 1]:
                            initial[r][i] = new_val
        
        try:
            improved_rows, improved_max = simulated_annealing(initial, deadline_for_this)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Final validation
    if not validate(best_rows):
        # Fallback to greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
