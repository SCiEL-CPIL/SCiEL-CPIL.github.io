import random
import time
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s. We use a safety margin.
    deadline = start_time + 140.0
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            row_diffs = get_row_diffs_set(row)
            if len(row_diffs) != diffs_per_row:
                return False
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True

    def greedy_row(max_val, global_used_set):
        """
        Greedily construct a row of length k+1 starting with 0, 
        with all values <= max_val, avoiding conflicts with global_used_set.
        """
        row = [0]
        current_diffs = set()
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            # Try smallest possible next value
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in current_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                if new_diffs & global_used_set:
                    continue
                
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            
            if not found:
                return None
        return row

    def random_row(max_val, global_used_set, attempts=1):
        """
        Try to find a valid row with max <= max_val using randomized greedy.
        """
        for _ in range(attempts):
            row = [0]
            current_diffs = set()
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = set()
                    for u in row:
                        d = v - u
                        if d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    if new_diffs & global_used_set:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                return row
        return None

    # 1. Initial Solution: Greedy with large scope
    init_scope = 300
    rows = [None] * n_rows
    success = False
    
    # Try to find a solution with backtracking within a reasonable time
    # We use a simple greedy fill first to get a valid starting point for SA
    global_used = set()
    current_rows = []
    for i in range(n_rows):
        row = greedy_row(init_scope, global_used)
        if row is None:
            # Fallback: try random
            row = random_row(init_scope, global_used, attempts=5)
        if row is None:
            # Last resort: very large scope
            row = greedy_row(1000, global_used)
        current_rows.append(row)
        global_used |= get_row_diffs_set(row)
        
    # Ensure validity of initial rows
    if not is_valid(current_rows):
        # If greedy failed to produce valid rows (unlikely with large scope but possible if logic bug),
        # force a valid construction with even larger scope
        global_used = set()
        current_rows = []
        for i in range(n_rows):
            row = greedy_row(1000, global_used)
            if row is None:
                # This should never happen for reasonable n_rows, k
                row = [0] + list(range(1, k+1))
            current_rows.append(row)
            global_used |= get_row_diffs_set(row)

    best_rows = [list(r) for r in current_rows]
    best_max = get_max_entry(best_rows)
    
    # 2. Simulated Annealing Optimization
    # Objective: Minimize soft-max of row maxima
    # soft_max(x, T) = T * log(sum(exp(x_i/T)))
    # As T -> 0, soft_max -> max.
    
    def soft_max(values, T):
        if T <= 1e-6:
            return max(values)
        # Numerical stability
        max_val = max(values)
        if max_val == 0:
            return 0
        # log(sum(exp(v/T))) = max/T + log(sum(exp((v-max)/T)))
        shifted = [v - max_val for v in values]
        exp_sum = sum(math.exp(v/T) for v in shifted)
        return max_val + T * math.log(exp_sum)

    def objective(rows, T):
        maxes = [max(row) for row in rows]
        return soft_max(maxes, T)

    def get_diffs_except_row(rows, idx):
        used = set()
        for i in range(n_rows):
            if i != idx:
                used |= get_row_diffs_set(rows[i])
        return used

    def apply_move(rows, row_idx, move):
        """
        Apply a move to a copy of the row.
        move: (type, params)
        type 'shift': (pos, delta)
        type 'swap': (pos1, pos2)
        Returns new_row or None if invalid (non-increasing or 0 constraint)
        """
        row = rows[row_idx][:]
        m_type, p1, p2 = move
        
        if m_type == 'shift':
            pos, delta = p1, p2
            old_val = row[pos]
            new_val = old_val + delta
            if new_val < 0:
                return None
            if pos == 0 and new_val != 0:
                return None
            # Check increasing order
            if pos > 0 and new_val <= row[pos-1]:
                return None
            if pos < len(row)-1 and new_val >= row[pos+1]:
                return None
            row[pos] = new_val
            return row
        elif m_type == 'swap':
            pos1, pos2 = p1, p2
            if pos1 == pos2:
                return row
            if pos1 == 0 or pos2 == 0:
                return None # Can't swap 0
            v1, v2 = row[pos1], row[pos2]
            row[pos1], row[pos2] = v2, v1
            # Check increasing order
            for i in range(len(row)-1):
                if row[i] >= row[i+1]:
                    return None
            return row
        return None

    def is_row_valid_with_global(new_row, global_used_except):
        row_diffs = get_row_diffs_set(new_row)
        if len(row_diffs) != diffs_per_row:
            return False
        if row_diffs & global_used_except:
            return False
        return True

    current_obj = objective(current_rows, 1.0)
    T_start = max(1.0, best_max / 10.0)
    T_end = 0.01
    cooling_rate = 0.999
    
    # Precompute global used for each row to speed up moves?
    # In SA, we modify one row at a time.
    # We maintain `current_rows` and `global_used_except[i]` might be expensive to recompute every step.
    # Instead, we check validity of the new row against the *current* set of diffs of other rows.
    # Since we only change one row, we can compute `used_others` once per iteration or maintain it.
    # For simplicity and safety, we recompute `used_others` only when accepting a move, 
    # or we can compute it on the fly. Given n_rows=4, k=8, it's fast.
    
    t = start_time
    step = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    # Initial temperature
    T = T_start
    
    while time.time() < deadline - 5.0:
        # Pick a random row
        row_idx = random.randint(0, n_rows - 1)
        
        # Generate a move
        # 80% shift, 20% swap
        if random.random() < 0.8:
            # Shift a random position by a small delta
            pos = random.randint(1, k) # Index 1 to k (0 is fixed)
            # Delta range: smaller as T decreases? Or fixed?
            # Let's use a range proportional to T or fixed small range
            delta_range = max(1, int(T))
            delta = random.randint(-delta_range, delta_range)
            if delta == 0:
                continue
            move = ('shift', pos, delta)
        else:
            # Swap two positions
            pos1 = random.randint(1, k)
            pos2 = random.randint(1, k)
            if pos1 == pos2:
                continue
            move = ('swap', pos1, pos2)
            
        new_row = apply_move(current_rows, row_idx, move)
        if new_row is None:
            continue
            
        # Check validity against other rows
        # Compute used diffs of other rows
        used_others = set()
        for i in range(n_rows):
            if i != row_idx:
                used_others |= get_row_diffs_set(current_rows[i])
                
        if not is_row_valid_with_global(new_row, used_others):
            continue
            
        # Calculate new objective
        # We need to update the maxes for soft_max
        old_max = max(current_rows[row_idx])
        new_max = max(new_row)
        
        # If the max didn't change and it wasn't the unique max, objective might not change much?
        # But soft-max is sensitive to all values if T is large.
        # To be precise, we should recalculate soft-max.
        old_maxes = [max(r) for r in current_rows]
        new_maxes = old_maxes[:]
        new_maxes[row_idx] = new_max
        
        old_obj = objective(current_rows, T) # This is O(n_rows * k^2) due to max calc? No, max is O(k).
        # Wait, objective calls soft_max which is O(n_rows).
        # But we already have old_maxes.
        # Let's just compute new_obj directly from new_maxes.
        
        # Optimization: We can compute old_obj once per loop? No, it depends on T.
        # Let's just compute both.
        # Actually, we can store current_obj.
        new_obj = soft_max(new_maxes, T)
        
        delta_obj = new_obj - current_obj
        
        accept = False
        if delta_obj < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_obj / T)
            if T > 1e-6:
                prob = math.exp(-delta_obj / T)
                if random.random() < prob:
                    accept = True
            # If T is very small, only accept improvements (or equal)
            elif delta_obj == 0:
                accept = True

        if accept:
            current_rows[row_idx] = new_row
            current_obj = new_obj
            
            # Update best if valid (it is valid by construction)
            curr_max = get_max_entry(current_rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [list(r) for r in current_rows]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
        else:
            no_improve_steps += 1
            
        # If stuck for too long, try a larger perturbation or restart?
        # For now, just continue. The random delta range increases with T, so it will explore.
        
        # Time check
        if time.time() > deadline - 2.0:
            break

    # 3. Local Search Polish (Greedy Shrink)
    # Use the SA result as a starting point for greedy shrinking
    # This is similar to the previous local search but starts from a potentially better state.
    
    # We'll do a few passes of trying to shrink the max of each row.
    for _ in range(10): # Max 10 passes
        if time.time() > deadline - 1.0:
            break
        improved = False
        for row_idx in range(n_rows):
            if time.time() > deadline - 1.0:
                break
            curr_row_max = current_rows[row_idx][-1]
            if curr_row_max <= lower_bound:
                continue
            
            # Try to find a row with max < curr_row_max
            target = curr_row_max - 1
            used_others = set()
            for i in range(n_rows):
                if i != row_idx:
                    used_others |= get_row_diffs_set(current_rows[i])
            
            new_row = random_row(target, used_others, attempts=5)
            if new_row is not None:
                # Validate
                if is_row_valid_with_global(new_row, used_others):
                    current_rows[row_idx] = new_row
                    improved = True
                    curr_max = get_max_entry(current_rows)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = [list(r) for r in current_rows]
        
        if not improved:
            break

    # Final Validation
    if not is_valid(best_rows):
        # Fallback to greedy if SA/Local search corrupted something (shouldn't happen, but safety)
        global_used = set()
        best_rows = []
        for i in range(n_rows):
            row = greedy_row(1000, global_used)
            if row is None:
                row = [0] + list(range(1, k+1))
            best_rows.append(row)
            global_used |= get_row_diffs_set(row)

    return best_rows
