import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def compute_scope(rows):
        return max(max(r) for r in rows)

    def greedy_construct(seed=None, scope_limit=100000):
        """
        Greedy construction with randomization.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                found = False
                # Try a window of candidates, occasionally skip ahead
                # to allow more flexibility
                for x in range(last_val + 1, min(last_val + 200, scope_limit) + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    # Try a wider search
                    for x in range(last_val + 1, scope_limit + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            for y in row[:-1]:
                                used_diffs.add(x - y)
                            found = True
                            break
                if not found:
                    return None
            rows.append(row)
        return rows

    def backtracking_init(time_limit=5.0):
        """
        Backtracking with forward-checking: build all rows simultaneously,
        choosing elements in a round-robin fashion across rows.
        This allows sacrificing an early entry to enable smaller later entries.
        """
        t_start = time.time()
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # We'll fill elements in order: for each "level" pos from 1 to k,
        # fill row 0's pos, then row 1's pos, etc.
        # But to allow more flexibility, we can try different orderings.
        
        # Simple approach: fill each row completely, but with backtracking
        # that can go back to previous rows.
        
        # For simplicity and speed, we'll do a depth-first search with
        # limited backtracking depth.
        
        # Actually, let's do a simpler thing: try multiple random greedy
        # constructions and pick the best.
        
        best_rows = None
        best_scope = float('inf')
        
        num_trials = 20
        for trial in range(num_trials):
            if time.time() - t_start >= time_limit:
                break
            r = greedy_construct(seed=trial)
            if r is not None:
                s = compute_scope(r)
                if s < best_scope:
                    best_scope = s
                    best_rows = r
        
        return best_rows

    # Step 1: Initialization
    # Try multiple random greedy constructions
    best_rows = None
    best_scope = float('inf')
    
    init_time_limit = min(10.0, deadline - start_time - 50.0)
    t_init_start = time.time()
    
    # Try a few seeds
    for seed in range(10):
        if time.time() - t_init_start >= init_time_limit:
            break
        r = greedy_construct(seed=seed)
        if r is not None:
            s = compute_scope(r)
            if s < best_scope:
                best_scope = s
                best_rows = r
    
    if best_rows is None:
        # Fallback: simple greedy without randomization
        best_rows = greedy_construct()
        if best_rows is None:
            # Last resort
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            best_rows = rows
    
    current_scope = compute_scope(best_rows)

    # Step 2: Simulated annealing
    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a byte array for O(1) difference lookups.
        Supports single-element moves and occasional 2-element moves.
        """
        rows = [r[:] for r in initial_rows]
        
        # Estimate max possible difference for array size
        max_possible = max(max(r) for r in rows) + 200
        # Use a byte array for used differences
        used_arr = bytearray(max_possible + 1)
        
        # Initialize used_arr
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        used_arr[d] = 1
        
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Temperature schedule
        t_start = time.time()
        initial_temp = max(2.0, current_scope / 4.0)
        cooling_rate = 0.9998
        
        # Precompute for speed: for each row and position, the list of other positions
        # We'll compute diffs on the fly
        
        def try_move(row_idx, pos, new_val):
            """Try to set rows[row_idx][pos] to new_val. Returns True if valid."""
            old_val = rows[row_idx][pos]
            if new_val == old_val:
                return False
            if pos > 0 and new_val <= rows[row_idx][pos-1]:
                return False
            if pos < k and new_val >= rows[row_idx][pos+1]:
                return False
            
            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)
            
            # Temporarily remove old_diffs from used_arr
            for d in old_diffs:
                if d <= max_possible:
                    used_arr[d] = 0
            
            # Try adding new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d > max_possible:
                        valid = False
                        break
                    if used_arr[d] == 1:
                        valid = False
                        break
                    new_diffs.append(d)
            
            if not valid:
                # Restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                return False
            
            # Compute new scope
            new_row = rows[row_idx][:]
            new_row[pos] = new_val
            new_scope = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
            new_scope = max(new_scope, max(new_row))
            
            return new_diffs, new_scope, old_diffs

        while time.time() - t_start < time_limit:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Bias toward the last element to encourage scope reduction
            if random.random() < 0.5:
                pos = k  # Last element
            else:
                pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            # Try to change this value
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 100
            
            if min_val > max_val:
                continue

            # For the last element, bias toward smaller values to reduce scope
            if pos == k and random.random() < 0.7:
                # Try values in [min_val, old_val] to shrink
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val)
                else:
                    continue
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue

            result = try_move(row_idx, pos, new_val)
            if result is False:
                continue
            
            new_diffs, new_scope, old_diffs = result
            
            # Accept or reject using smooth surrogate
            delta = current_scope - new_scope
            if delta > 0 or random.random() < math.exp(delta / max(initial_temp, 0.1)):
                # Accept
                rows[row_idx][pos] = new_val
                for d in new_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope

    def local_polish(rows, time_limit):
        """
        Final high-precision polish: try to reduce each element by scanning downward.
        """
        rows = [r[:] for r in rows]
        
        # Build used differences set
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_set.add(row[j] - row[i])
        
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            improved = False
            
            for row_idx in range(n_rows):
                if time.time() - t_start >= time_limit:
                    break
                
                row = rows[row_idx]
                
                # Try to reduce each position from last to first
                for pos in range(k, 0, -1):
                    if time.time() - t_start >= time_limit:
                        break
                    
                    current_val = row[pos]
                    min_allowed = row[pos-1] + 1
                    max_allowed = row[pos+1] - 1 if pos < k else current_val
                    
                    # Scan downward from current_val - 1
                    for new_val in range(current_val - 1, min_allowed - 1, -1):
                        if new_val > max_allowed:
                            continue
                        
                        # Check if we can set row[pos] to new_val
                        # Remove old diffs involving position pos
                        old_diffs = set()
                        for i in range(k + 1):
                            if i != pos:
                                if i < pos:
                                    d = current_val - row[i]
                                else:
                                    d = row[i] - current_val
                                old_diffs.add(d)
                        
                        # Check new diffs
                        valid = True
                        new_diffs = []
                        for i in range(k + 1):
                            if i != pos:
                                if i < pos:
                                    d = new_val - row[i]
                                else:
                                    d = row[i] - new_val
                                if d in used_set or d in old_diffs:
                                    valid = False
                                    break
                                new_diffs.append(d)
                        
                        if valid:
                            # Apply change
                            for d in old_diffs:
                                used_set.discard(d)
                            for d in new_diffs:
                                used_set.add(d)
                            
                            row[pos] = new_val
                            current_scope = compute_scope(rows)
                            improved = True
                            
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                            
                            # Continue trying to reduce further
                            current_val = new_val
                            max_allowed = row[pos+1] - 1 if pos < k else new_val
                        else:
                            break
            
            if not improved:
                break
        
        return best_rows, best_scope

    # Step 2: Simulated annealing to improve
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Split time between SA and polish
        sa_time = remaining_time * 0.65
        polish_time = remaining_time * 0.30 - 5
        
        best_rows, best_scope = simulated_annealing(best_rows, sa_time)
        
        # Step 3: Local polish
        remaining = deadline - time.time()
        if remaining > 5:
            final_rows, final_scope = local_polish(best_rows, min(polish_time, remaining - 3))
            return final_rows
        
        return best_rows

    return best_rows
