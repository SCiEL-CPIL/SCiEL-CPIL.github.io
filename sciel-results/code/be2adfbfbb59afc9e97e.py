import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    k_plus_1 = k + 1
    
    # Helper to compute the used differences mask for a list of rows
    def compute_used_mask(rows):
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used |= (1 << d)
        return used

    def get_max(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if all positive differences across all rows are distinct."""
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if (used >> d) & 1:
                        return False
                    used |= (1 << d)
        return True

    # 1. Greedy Initial Solution
    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = get_max(best_rows)
    
    # 2. Simulated Annealing
    # We represent a solution as a flat list of all entries.
    # Moves: 
    #   - Pick a random row and a random position (not 0), try to change its value
    #   - Or pick two elements in different rows and swap their values (if it keeps ordering)
    #   - Or pick an element and shift it by a small random amount
    
    def simulated_annealing(initial_rows, max_time):
        rows = [list(r) for r in initial_rows]
        current_max = get_max(rows)
        best_rows_sa = [list(r) for r in rows]
        best_max_sa = current_max
        
        # Temperature schedule
        t_initial = 50.0
        t_final = 0.1
        cooling_rate = 0.995
        
        def get_cost(rows):
            """Cost = max entry + penalty for invalidity"""
            m = get_max(rows)
            # Check validity
            used = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if (used >> d) & 1:
                            return m + 1000  # Large penalty for invalid
                        used |= (1 << d)
            return m
        
        current_cost = get_cost(rows)
        
        def make_move(rows, t):
            """Make a random move to the rows. Returns (new_rows, old_value, new_value, row_idx, pos_idx)"""
            rows_copy = [list(r) for r in rows]
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.6:
                # Change a single element
                row_idx = random.randint(0, n_rows - 1)
                pos_idx = random.randint(1, len(rows_copy[row_idx]) - 1)  # Not position 0
                
                old_val = rows_copy[row_idx][pos_idx]
                # Determine valid range
                min_val = rows_copy[row_idx][pos_idx - 1] + 1
                max_val = rows_copy[row_idx][pos_idx + 1] - 1 if pos_idx < len(rows_copy[row_idx]) - 1 else current_max + 20
                
                # Random offset
                offset_range = int(t * 3) + 1
                new_val = old_val + random.randint(-offset_range, offset_range)
                new_val = max(min_val, min(max_val, new_val))
                
                if new_val != old_val:
                    rows_copy[row_idx][pos_idx] = new_val
                    return rows_copy, old_val, new_val, row_idx, pos_idx
            
            elif move_type < 0.8:
                # Swap two elements in different rows
                r1 = random.randint(0, n_rows - 1)
                r2 = random.randint(0, n_rows - 1)
                if r1 == r2:
                    return None, None, None, None, None
                
                p1 = random.randint(1, len(rows_copy[r1]) - 1)
                p2 = random.randint(1, len(rows_copy[r2]) - 1)
                
                v1 = rows_copy[r1][p1]
                v2 = rows_copy[r2][p2]
                
                # Check if swap maintains ordering
                # For row r1: need rows_copy[r1][p1-1] < v2 < rows_copy[r1][p1+1]
                if p1 > 0 and v2 <= rows_copy[r1][p1-1]:
                    return None, None, None, None, None
                if p1 < len(rows_copy[r1]) - 1 and v2 >= rows_copy[r1][p1+1]:
                    return None, None, None, None, None
                # For row r2: need rows_copy[r2][p2-1] < v1 < rows_copy[r2][p2+1]
                if p2 > 0 and v1 <= rows_copy[r2][p2-1]:
                    return None, None, None, None, None
                if p2 < len(rows_copy[r2]) - 1 and v1 >= rows_copy[r2][p2+1]:
                    return None, None, None, None, None
                
                rows_copy[r1][p1] = v2
                rows_copy[r2][p2] = v1
                return rows_copy, v1, v2, r1, p1
            
            else:
                # Shift a whole row segment
                row_idx = random.randint(0, n_rows - 1)
                pos_idx = random.randint(1, len(rows_copy[row_idx]) - 1)
                shift = random.randint(-5, 5)
                if shift == 0:
                    return None, None, None, None, None
                
                # Apply shift to this and all subsequent elements
                for p in range(pos_idx, len(rows_copy[row_idx])):
                    rows_copy[row_idx][p] += shift
                
                # Check ordering
                for p in range(1, len(rows_copy[row_idx])):
                    if rows_copy[row_idx][p] <= rows_copy[row_idx][p-1]:
                        return None, None, None, None, None
                
                # Check non-negative
                if rows_copy[row_idx][0] != 0:
                    return None, None, None, None, None
                
                return rows_copy, None, None, row_idx, pos_idx
            
            return None, None, None, None, None
        
        t = t_initial
        iterations = 0
        
        while time.time() - start_time < max_time and t > t_final:
            new_rows, old_val, new_val, row_idx, pos_idx = make_move(rows, t)
            
            if new_rows is None:
                t *= cooling_rate
                iterations += 1
                continue
            
            new_cost = get_cost(new_rows)
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta < 0 or random.random() < pow(2.718281828, -delta / t):
                rows = new_rows
                current_cost = new_cost
                
                if current_cost < best_max_sa:
                    best_max_sa = current_cost
                    best_rows_sa = [list(r) for r in rows]
            
            t *= cooling_rate
            iterations += 1
        
        return best_rows_sa, best_max_sa
    
    # Run simulated annealing with multiple restarts
    sa_time_budget = min(60, timeout - 5)  # Leave time for final polish
    sa_start = time.time()
    
    for restart in range(5):
        if time.time() - start_time > sa_start + sa_time_budget:
            break
        
        # Perturb the current best to create diversity
        perturbed = [list(r) for r in best_rows]
        for _ in range(3):
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, len(perturbed[r_idx]) - 1)
            perturbed[r_idx][p_idx] += random.randint(-10, 10)
            # Re-sort and ensure validity of ordering
            perturbed[r_idx].sort()
            # Ensure starts at 0
            if perturbed[r_idx][0] != 0:
                offset = perturbed[r_idx][0]
                perturbed[r_idx] = [x - offset for x in perturbed[r_idx]]
        
        sa_rows, sa_max = simulated_annealing(perturbed, sa_start + sa_time_budget - (time.time() - sa_start))
        
        if sa_max < best_max:
            best_rows = sa_rows
            best_max = sa_max
    
    # 3. Local Search Polish
    def local_search_polish(rows):
        rows = [list(r) for r in rows]
        used = compute_used_mask(rows)
        current_max = get_max(rows)
        improved = True
        
        while improved and time.time() - start_time < timeout:
            improved = False
            # Find the global maximum element
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_val <= 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Determine valid range for replacement
            min_val = row[max_i - 1] + 1 if max_i > 0 else 1
            max_allowed = current_val - 1
            if max_i < len(row) - 1:
                max_allowed = min(max_allowed, row[max_i + 1] - 1)
            
            if max_allowed < min_val:
                continue
            
            # Calculate old differences involving current_val
            old_diffs = 0
            for a in row:
                if a != current_val:
                    d = abs(a - current_val)
                    old_diffs |= (1 << d)
            
            used_without_old = used & ~old_diffs
            
            # Try to find a smaller valid value v
            found_replacement = False
            for v in range(min_val, max_allowed + 1):
                valid = True
                new_diffs = 0
                for a in row:
                    if a != current_val:
                        d = abs(a - v)
                        if (used_without_old >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                
                if valid:
                    row[max_i] = v
                    used = used_without_old | new_diffs
                    improved = True
                    found_replacement = True
                    break
            
            if not found_replacement:
                break

        return rows, get_max(rows)

    polished_rows, polished_max = local_search_polish(best_rows)
    if polished_max < best_max:
        best_rows = polished_rows
        best_max = polished_max

    # Final validity check
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()
        best_max = get_max(best_rows)

    return best_rows
# EVOLVE-BLOCK-END
