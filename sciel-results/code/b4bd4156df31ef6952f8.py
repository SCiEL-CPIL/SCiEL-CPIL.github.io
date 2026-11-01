import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 132s to be safe.
    deadline = start_time + 132.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        Optimized: precompute row elements for O(1) diff checks.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                row_set = set(row)
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def local_search(rows, time_limit):
        """
        Iterated Local Search (ILS) with Targeted Descent, Two-Element Swaps, and Row Reconstruction.
        1. Try to reduce the global maximum via single-element moves.
        2. If stuck, try two-element swaps within the same row to escape local minima.
        3. If stuck longer, reconstruct a random row entirely.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        def try_reduce_element(r_idx, c_idx, max_search=200):
            """Try to reduce element at (r_idx, c_idx). Returns new scope if improved, else None."""
            row = best_rows[r_idx]
            current_val = row[c_idx]
            prev_val = row[c_idx - 1]
            
            # Build set of all diffs
            all_diffs = get_all_diffs(best_rows)
            
            # Diffs to remove (involving target)
            remove_set = set()
            for i in range(c_idx):
                remove_set.add(current_val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            # Try values from current_val - 1 down to prev_val + 1
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left_diffs = [new_val - row[i] for i in range(c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                row[c_idx] = new_val
                return get_scope(best_rows)
            
            return None

        def try_two_element_swap(r_idx, c1, c2, max_attempts=50):
            """Try to swap/reassign values at positions c1 and c2 in row r_idx."""
            if c1 >= c2:
                return None
            row = best_rows[r_idx]
            v1, v2 = row[c1], row[c2]
            prev1 = row[c1-1]
            next2 = row[c2+1] if c2 < row_len-1 else v2 + 100
            
            # Build all diffs
            all_diffs = get_all_diffs(best_rows)
            
            # Diffs involving both elements
            remove_set = set()
            for i in range(len(row)):
                if i != c1 and i != c2:
                    d1 = abs(v1 - row[i])
                    d2 = abs(v2 - row[i])
                    if d1 > 0:
                        remove_set.add(d1)
                    if d2 > 0:
                        remove_set.add(d2)
            # Also remove the diff between c1 and c2
            remove_set.add(v2 - v1)
            
            fixed_diffs = all_diffs - remove_set
            
            # Try random reassignments
            for _ in range(max_attempts):
                # Try swapping values
                if random.random() < 0.3:
                    new_v1, new_v2 = v2, v1
                else:
                    # Try random values in range
                    low1 = prev1 + 1
                    high1 = (v2 - 1) if c2 == c1 + 1 else (min(v2 - 1, (v2 + v1)//2 - 1))
                    low2 = v1 + 1
                    high2 = next2 - 1
                    
                    if low1 > high1 or low2 > high2:
                        continue
                    
                    new_v1 = random.randint(low1, high1)
                    new_v2 = random.randint(low2, high2)
                    if new_v1 >= new_v2:
                        continue
                
                # Check validity
                valid = True
                for i in range(len(row)):
                    if i == c1 or i == c2:
                        continue
                    d1 = new_v1 - row[i]
                    d2 = new_v2 - row[i]
                    if d1 <= 0 or d2 <= 0:
                        valid = False
                        break
                    if d1 in fixed_diffs or d2 in fixed_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check diff between the two new values
                if new_v2 - new_v1 in fixed_diffs:
                    continue
                
                # Check internal uniqueness of new diffs
                new_diffs = set()
                for i in range(c1):
                    new_diffs.add(new_v1 - row[i])
                for i in range(c1+1, c2):
                    new_diffs.add(row[i] - new_v1)
                    new_diffs.add(new_v2 - row[i])
                for i in range(c2+1, len(row)):
                    new_diffs.add(new_v2 - row[i])
                
                if len(new_diffs) != len(new_diffs):
                    continue
                
                # Check against fixed_diffs
                conflict = False
                for d in new_diffs:
                    if d in fixed_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Apply
                row[c1] = new_v1
                row[c2] = new_v2
                return get_scope(best_rows)
            
            return None

        stuck_counter = 0
        no_improve_counter = 0

        while time.time() < start + time_limit:
            # 1. Targeted Descent: Try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            current_val = max_val
            found_improvement = False
            new_scope = try_reduce_element(max_r_idx, max_c_idx)
            
            if new_scope is not None:
                found_improvement = True
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                stuck_counter += 1
                no_improve_counter += 1
                
                # 2. Try two-element swap in the row with the max element
                if stuck_counter > 5 and random.random() < 0.4:
                    c1 = random.randint(1, max_c_idx - 1) if max_c_idx > 2 else 1
                    c2 = random.randint(c1 + 1, max_c_idx) if c1 < max_c_idx else max_c_idx
                    if c2 <= c1:
                        c2 = c1 + 1
                        if c2 >= row_len:
                            c2 = row_len - 1
                            c1 = c2 - 1
                    if c1 >= 1 and c2 < row_len and c1 < c2:
                        swap_scope = try_two_element_swap(max_r_idx, c1, c2)
                        if swap_scope is not None and swap_scope <= best_scope:
                            if swap_scope < best_scope:
                                best_scope = swap_scope
                                stuck_counter = 0
                                no_improve_counter = 0
                            else:
                                no_improve_counter += 1
                            continue
            
            # 3. If stuck for a while, try row reconstruction
            if (stuck_counter > 15 or no_improve_counter > 40) and random.random() < 0.3:
                r_idx = random.randint(0, n_rows - 1)
                
                other_rows = [best_rows[i] for i in range(n_rows) if i != r_idx]
                
                fixed_diffs = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            fixed_diffs.add(r[j] - r[i])
                
                success = False
                for attempt in range(8):
                    temp_row = [0]
                    temp_used = set()
                    
                    while len(temp_row) < row_len:
                        last = temp_row[-1]
                        candidates = []
                        x = last + 1
                        limit = min(best_scope, last + 60)
                        
                        while x <= limit and len(candidates) < 25:
                            valid = True
                            for a in temp_row:
                                d = x - a
                                if d in fixed_diffs or d in temp_used:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                            x += 1
                        
                        if not candidates:
                            while True:
                                x += 1
                                valid = True
                                for a in temp_row:
                                    d = x - a
                                    if d in fixed_diffs or d in temp_used:
                                        valid = False
                                        break
                                if valid:
                                    candidates.append(x)
                                    break
                        
                        if random.random() < 0.25 and len(candidates) > 1:
                            x = random.choice(candidates)
                        else:
                            x = candidates[0]
                        
                        temp_row.append(x)
                        for a in temp_row[:-1]:
                            temp_used.add(x - a)
                    
                    if max(temp_row) <= best_scope:
                        best_rows[r_idx] = temp_row
                        success = True
                        break
                
                if success:
                    stuck_counter = 0
                    no_improve_counter = 0
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                else:
                    # Standard perturbation: Move a random element
                    r_idx = random.randint(0, n_rows-1)
                    c_idx = random.randint(1, row_len-1)
                    
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx-1]
                    next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                    
                    all_diffs = get_all_diffs(best_rows)
                    remove_set = set()
                    row = best_rows[r_idx]
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                    
                    fixed_diffs = all_diffs - remove_set
                    
                    low = prev_v + 1
                    high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                    
                    if low <= high:
                        applied = False
                        for _ in range(15):
                            cand = random.randint(low, high)
                            if cand == curr_val:
                                continue
                            
                            valid = True
                            for i in range(c_idx):
                                d = cand - row[i]
                                if d in fixed_diffs:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            for j in range(c_idx + 1, len(row)):
                                d = row[j] - cand
                                if d in fixed_diffs:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            new_left = [cand - row[i] for i in range(c_idx)]
                            new_right = [row[j] - cand for j in range(c_idx+1, len(row))]
                            if len(set(new_left + new_right)) != len(new_left + new_right):
                                continue
                            
                            row[c_idx] = cand
                            applied = True
                            break
                        
                        if applied:
                            new_scope = get_scope(best_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                stuck_counter = 0
                                no_improve_counter = 0
                            else:
                                no_improve_counter += 1

        return best_rows, best_scope

    def polish_pass(rows, time_limit):
        """
        Final high-precision polish pass: iteratively try to reduce each element,
        including two-element swaps to escape local minima.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def get_all_diffs(rws):
            diffs = set()
            for r in rws:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs
        
        def try_reduce(r_idx, c_idx):
            """Try to reduce element at (r_idx, c_idx). Returns True if reduced."""
            row = best_rows[r_idx]
            val = row[c_idx]
            prev_val = row[c_idx - 1]
            
            all_diffs = get_all_diffs(best_rows)
            
            remove_set = set()
            for i in range(c_idx):
                remove_set.add(val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - val)
            
            fixed_diffs = all_diffs - remove_set
            
            for new_val in range(val - 1, prev_val, -1):
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left = [new_val - row[i] for i in range(c_idx)]
                new_right = [row[j] - new_val for j in range(c_idx+1, len(row))]
                if len(set(new_left + new_right)) != len(new_left + new_right):
                    continue
                
                row[c_idx] = new_val
                return True
            
            return False

        def try_swap(r_idx, c1, c2):
            """Try to swap/reassign values at positions c1 and c2 in row r_idx."""
            if c1 >= c2:
                return False
            row = best_rows[r_idx]
            v1, v2 = row[c1], row[c2]
            
            all_diffs = get_all_diffs(best_rows)
            
            remove_set = set()
            for i in range(len(row)):
                if i != c1 and i != c2:
                    d1 = abs(v1 - row[i])
                    d2 = abs(v2 - row[i])
                    if d1 > 0:
                        remove_set.add(d1)
                    if d2 > 0:
                        remove_set.add(d2)
            remove_set.add(v2 - v1)
            
            fixed_diffs = all_diffs - remove_set
            
            # Try a few random reassignments
            for _ in range(20):
                if random.random() < 0.2:
                    new_v1, new_v2 = v2, v1
                else:
                    prev1 = row[c1-1]
                    next2 = row[c2+1] if c2 < row_len-1 else v2 + 100
                    low1 = prev1 + 1
                    high1 = v2 - 1
                    low2 = v1 + 1
                    high2 = next2 - 1
                    
                    if low1 > high1 or low2 > high2:
                        continue
                    new_v1 = random.randint(low1, high1)
                    new_v2 = random.randint(low2, high2)
                    if new_v1 >= new_v2:
                        continue
                
                valid = True
                for i in range(len(row)):
                    if i == c1 or i == c2:
                        continue
                    d1 = new_v1 - row[i]
                    d2 = new_v2 - row[i]
                    if d1 <= 0 or d2 <= 0:
                        valid = False
                        break
                    if d1 in fixed_diffs or d2 in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                if new_v2 - new_v1 in fixed_diffs:
                    continue
                
                new_diffs = set()
                for i in range(c1):
                    new_diffs.add(new_v1 - row[i])
                for i in range(c1+1, c2):
                    new_diffs.add(row[i] - new_v1)
                    new_diffs.add(new_v2 - row[i])
                for i in range(c2+1, len(row)):
                    new_diffs.add(new_v2 - row[i])
                
                if len(new_diffs) != len(new_diffs):
                    continue
                
                conflict = False
                for d in new_diffs:
                    if d in fixed_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                row[c1] = new_v1
                row[c2] = new_v2
                return True
            
            return False

        while time.time() < start + time_limit:
            elements = []
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if c_idx > 0:
                        elements.append((val, r_idx, c_idx))
            
            if not elements:
                break
            
            elements.sort(reverse=True)
            
            improved = False
            for val, r_idx, c_idx in elements:
                if time.time() > start + time_limit:
                    break
                
                if try_reduce(r_idx, c_idx):
                    improved = True
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    break
            
            if not improved:
                # Try some swaps
                for _ in range(5):
                    if time.time() > start + time_limit:
                        break
                    r_idx = random.randint(0, n_rows-1)
                    c1 = random.randint(1, row_len-2)
                    c2 = random.randint(c1+1, row_len-1)
                    if try_swap(r_idx, c1, c2):
                        improved = True
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break
            
            if not improved:
                break
        
        return best_rows, best_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Run several greedy initializations with different seeds
    num_inits = 12
    init_time_limit = 6.0
    start_init = time.time()
    
    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.08 + 0.01*i, window=25, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Local Search Refinement
    local_time = 70.0
    if time.time() > deadline - 80:
        local_time = deadline - time.time() - 12
        
    if local_time > 15 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 2.5: Polish pass for final optimization
    polish_time = min(25.0, max(0.0, deadline - time.time() - 18.0))
    if polish_time > 8.0 and best_rows:
        polished_rows, polished_scope = polish_pass(best_rows, polish_time)
        if validate(polished_rows):
            best_rows = polished_rows
            best_scope = polished_scope

    # Phase 3: Backtracking to try to beat the local search result
    current_target = best_scope - 1
    seed_counter = 0
    
    # Reserve time for backtracking
    backtracking_time_remaining = deadline - time.time() - 12.0
    if backtracking_time_remaining < 10.0:
        return best_rows
    
    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking with candidate windowing.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == row_len:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            candidates = []
            # Scan a window first
            scan_limit = min(upper, lower + 50)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Run backtracking attempts
    while time.time() < deadline - 6:
        if current_target < 112:
            break
        
        remaining = deadline - time.time() - 6.0
        if remaining < 1.0:
            break
            
        search_time = min(6.0, remaining)
        if search_time < 1.0:
            break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1
        
        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    break
            else:
                break
        else:
            retry_remaining = deadline - time.time() - 6.0
            if retry_remaining > 1.0:
                result_retry = backtracking_search(n_rows, k, current_target, min(6.0, retry_remaining), seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
