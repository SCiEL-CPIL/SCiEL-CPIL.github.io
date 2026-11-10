import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Parameters
    # n_rows = 4, k = 7 => 8 elements per row, 28 diffs per row, 112 total diffs
    # Target: Minimize max element.
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def calculate_scope(rows):
        return max(max(r) for r in rows)

    def is_valid_and_scope(rows):
        """
        Returns (is_valid, scope, conflicts_count, diffs_set)
        """
        used_diffs = set()
        valid = True
        scope = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        conflicts += 1
                    else:
                        used_diffs.add(d)
            if max(row) > scope:
                scope = max(row)
        return valid, scope, conflicts, used_diffs

    def get_cost(rows):
        """
        Cost = scope if valid, else scope + penalty * conflicts
        """
        valid, scope, conflicts, _ = is_valid_and_scope(rows)
        if valid:
            return scope
        else:
            return scope + 100 * conflicts

    # Better Greedy: Strict sequential, but with randomized candidate selection
    def greedy_construct():
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

    # Randomized Greedy: Pick from a small set of valid candidates to diversify
    def greedy_random_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Look ahead for a few valid candidates
                candidates = []
                search_limit = x + 15
                while x < search_limit:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    # If we can't find any in the window, expand window
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    else:
                        # This shouldn't happen often if we keep expanding
                        # But if it does, we might get stuck.
                        # For now, just force the next valid one by scanning further
                        while True:
                            x += 1
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used_diffs):
                                candidates.append(x)
                                break
                
                # Pick a random candidate, biased towards smaller ones
                # Use a simple weighting: 1/(index+1)
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                total_w = sum(weights)
                r = random.random() * total_w
                acc = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen_idx = i
                        break
                
                chosen_val = candidates[chosen_idx]
                row.append(chosen_val)
                used_diffs.update({chosen_val - a for a in row[:-1]})
                x = chosen_val - 1  # Next search starts from here
                
                # If we picked a larger value, x might lag. Let's just keep x as the last chosen value.
                # Actually, the logic above sets x = chosen_val - 1, so next loop x becomes chosen_val.
                # This is fine.
            
            rows.append(row)
        return rows

    # Initialize best
    best_rows = greedy_construct()
    valid, best_scope, _, _ = is_valid_and_scope(best_rows)
    if not valid:
        # Fallback, shouldn't happen for greedy
        return None

    best_valid = True
    best_cost = best_scope
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    current_valid = True
    current_conflicts = 0
    
    current_cost = get_cost(current_rows)
    
    T = 50.0
    alpha = 0.997
    min_T = 0.5
    steps = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 1:
        # Generate Move
        # Pick random row and index
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k) # 1 to k (0 is fixed)
        
        # Generate new value
        # Range of delta - use larger jumps for more exploration
        delta_range = random.randint(1, 15)
        delta = random.randint(-delta_range, delta_range)
        new_val = current_rows[r_idx][i_idx] + delta
        
        # Check bounds
        lower_bound = current_rows[r_idx][i_idx - 1] + 1
        if i_idx < k:
            upper_bound = current_rows[r_idx][i_idx + 1] - 1
        else:
            upper_bound = current_rows[r_idx][i_idx] + 30 # Allow expansion
        
        if new_val < lower_bound or new_val > upper_bound:
            # Try to snap to boundary or skip
            # For now, skip this move
            continue

        # Apply move
        old_val = current_rows[r_idx][i_idx]
        current_rows[r_idx][i_idx] = new_val
        
        # Calculate new cost
        new_valid, new_scope, new_conflicts, _ = is_valid_and_scope(current_rows)
        new_cost = get_cost(current_rows)
        
        # Delta
        delta_cost = new_cost - current_cost
        
        # Accept/Reject
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            current_scope = new_scope
            current_valid = new_valid
            current_conflicts = new_conflicts
            
            if current_valid and current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                no_improve_steps = 0
                # Optional: if we reach theoretical lower bound, stop
                # lower_bound = 112
                if best_scope <= 112:
                    break
            else:
                no_improve_steps += 1
        else:
            # Reject, revert
            current_rows[r_idx][i_idx] = old_val
            no_improve_steps += 1
        
        T *= alpha
        if T < min_T:
            # Reheat or stop?
            # Let's reheat occasionally to escape local minima
            T = 15.0
        
        steps += 1
        
        # Occasionally, try a "Row Regeneration" move
        if steps % 500 == 0 and random.random() < 0.1:
            reg_idx = random.randint(0, n_rows - 1)
            # Compute diffs of other rows
            other_diffs = set()
            for i, r in enumerate(current_rows):
                if i == reg_idx: continue
                other_diffs.update(get_row_diffs(r))
            
            # Generate new row using greedy_random logic but constrained by other_diffs
            new_row = [0]
            x = 0
            success = True
            while len(new_row) < k + 1:
                candidates = []
                search_limit = x + 15
                while x < search_limit:
                    x += 1
                    new_diffs_row = {x - a for a in new_row}
                    # Check against other_diffs
                    if not (new_diffs_row & other_diffs):
                        # Also check internal uniqueness
                        internal_ok = True
                        test_row = new_row + [x]
                        test_diffs = set()
                        for i in range(len(test_row)):
                            for j in range(i+1, len(test_row)):
                                dd = test_row[j] - test_row[i]
                                if dd in test_diffs:
                                    internal_ok = False
                                    break
                                test_diffs.add(dd)
                        if internal_ok:
                            candidates.append(x)
                            if len(candidates) >= 5:
                                break
                
                if not candidates:
                    # Expand search
                    while True:
                        x += 1
                        new_diffs_row = {x - a for a in new_row}
                        if not (new_diffs_row & other_diffs):
                            internal_ok = True
                            test_row = new_row + [x]
                            test_diffs = set()
                            for i in range(len(test_row)):
                                for j in range(i+1, len(test_row)):
                                    dd = test_row[j] - test_row[i]
                                    if dd in test_diffs:
                                        internal_ok = False
                                        break
                                    test_diffs.add(dd)
                            if internal_ok:
                                candidates.append(x)
                                break
                        if x > 500: # Safety break
                            success = False
                            break
                
                if not success:
                    break
                
                # Pick random candidate
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                total_w = sum(weights)
                r = random.random() * total_w
                acc = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen_idx = i
                        break
                
                chosen_val = candidates[chosen_idx]
                new_row.append(chosen_val)
                x = chosen_val - 1
            
            if success and len(new_row) == k + 1:
                # Check if this new row + others is valid
                old_row = current_rows[reg_idx]
                current_rows[reg_idx] = new_row
                new_valid, new_scope, new_conflicts, _ = is_valid_and_scope(current_rows)
                new_cost = get_cost(current_rows)
                
                if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
                    current_cost = new_cost
                    current_scope = new_scope
                    current_valid = new_valid
                    current_conflicts = new_conflicts
                    if current_valid and current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        no_improve_steps = 0
                        if best_scope <= 112:
                            break
                    else:
                        no_improve_steps += 1
                else:
                    current_rows[reg_idx] = old_row
        
        # If stuck for too long, try a full restart with different seed
        if no_improve_steps > max_no_improve:
            no_improve_steps = 0
            # Try a few random restarts
            for _ in range(5):
                # Alternate between standard and random greedy
                if random.random() < 0.5:
                    restart_rows = greedy_construct()
                else:
                    restart_rows = greedy_random_construct()
                
                r_valid, r_scope, r_conflicts, _ = is_valid_and_scope(restart_rows)
                r_cost = get_cost(restart_rows)
                if r_cost < current_cost:
                    current_rows = [r[:] for r in restart_rows]
                    current_cost = r_cost
                    current_scope = r_scope
                    current_valid = r_valid
                    current_conflicts = r_conflicts
                    if current_valid and current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= 112:
                            break
                    break
            T = 30.0  # Reheat

    return best_rows
