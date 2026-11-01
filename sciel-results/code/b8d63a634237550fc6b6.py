import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 132.0  # Safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
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

    def greedy_init(rand_prob=0.12, window=25, seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
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

    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    def local_search(rows, time_limit):
        """
        Iterated Local Search with targeted descent, random perturbation, and row reconstruction.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def try_decrease_element(best_rows, r_idx, c_idx):
            """Try to decrease a specific element. Returns True if improved."""
            curr_val = best_rows[r_idx][c_idx]
            prev_v = best_rows[r_idx][c_idx-1]
            
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            row = best_rows[r_idx]
            for i in range(c_idx):
                remove_set.add(curr_val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - curr_val)
            
            fixed_diffs = all_diffs - remove_set
            
            for new_val in range(curr_val - 1, prev_v, -1):
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
                new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                if len(set(new_left + new_right)) != len(new_left + new_right):
                    continue
                
                row[c_idx] = new_val
                return True
            return False

        def reconstruct_row(best_rows, r_idx):
            """Reconstruct a single row to escape local minima. Returns True if successful."""
            row = best_rows[r_idx][:]
            # Remove diffs for this row
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            for i in range(len(row)):
                for j in range(i):
                    remove_set.add(row[j] - row[i])
            fixed_diffs = all_diffs - remove_set
            
            # Rebuild row
            new_row = [0]
            valid = True
            for _ in range(row_len - 1):
                last = new_row[-1]
                found = False
                x = last + 1
                while x <= best_scope + 10:
                    ok = True
                    for a in new_row:
                        if (x - a) in fixed_diffs:
                            ok = False
                            break
                    if ok:
                        # Also check within new_row
                        for i in range(len(new_row)):
                            for j in range(i):
                                if new_row[i] - new_row[j] == x - a:
                                    pass # already checked via fixed_diffs? No, fixed_diffs doesn't include new_row internal diffs
                        # Actually we need to check against new_row internal diffs too
                        # Let's just do a simple check
                        internal_ok = True
                        for i in range(len(new_row)):
                            d = x - new_row[i]
                            for j in range(i):
                                if d == new_row[i] - new_row[j]:
                                    internal_ok = False
                                    break
                            if not internal_ok:
                                break
                        if internal_ok:
                            new_row.append(x)
                            found = True
                            break
                    x += 1
                if not found:
                    valid = False
                    break
            
            if valid and len(new_row) == row_len:
                best_rows[r_idx] = new_row
                return True
            return False

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
            
            found_improvement = False
            
            # Try to decrease the max element
            if try_decrease_element(best_rows, max_r_idx, max_c_idx):
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                found_improvement = True
            
            if found_improvement:
                continue

            # 2. Try to decrease other elements
            improved_any = False
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        break
                    if try_decrease_element(best_rows, r_idx, c_idx):
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            stuck_counter = 0
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
                        improved_any = True
                        break
                if improved_any:
                    break
            
            if improved_any:
                continue

            # 3. Perturbation: Randomly move an element
            if time.time() < start + time_limit:
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                row = best_rows[r_idx]
                curr_val = row[c_idx]
                prev_v = row[c_idx-1]
                next_v = row[c_idx+1] if c_idx < row_len-1 else curr_val + 50
                
                all_diffs = get_all_diffs(best_rows)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set
                
                # Try a random new value
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
                
                if low <= high:
                    for _ in range(10):
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
                        new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        
                        row[c_idx] = cand
                        improved_any = True
                        break
                
                if improved_any:
                    continue

            # 4. Row Reconstruction if stuck
            if no_improve_counter > 5 or stuck_counter > 10:
                r_idx = random.randint(0, n_rows - 1)
                if reconstruct_row(best_rows, r_idx):
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        stuck_counter = 0
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                    continue
            
            no_improve_counter += 1
            stuck_counter += 1

        return best_rows, best_scope

    def mrv_backtracking(n_rows, k, target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()
        
        rows = [[-1] * row_len for _ in range(n_rows)]
        for r in range(n_rows):
            rows[r][0] = 0
        
        used_diffs = set()
        diff_to_positions = {}
        
        def add_diff(d, r, c):
            used_diffs.add(d)
            if d not in diff_to_positions:
                diff_to_positions[d] = set()
            diff_to_positions[d].add((r, c))
        
        def remove_diff(d, r, c):
            used_diffs.discard(d)
            if d in diff_to_positions:
                diff_to_positions[d].discard((r, c))
                if not diff_to_positions[d]:
                    del diff_to_positions[d]
        
        def get_candidates(r, c, max_val):
            if rows[r][c] != -1:
                return [rows[r][c]]
            
            prev_val = rows[r][c-1] if c > 0 else -1
            lower = prev_val + 1
            upper = min(max_val, target_scope)
            
            if lower > upper:
                return []
            
            candidates = []
            scan_limit = min(upper, lower + 80)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(c):
                    if rows[r][i] == -1:
                        continue
                    d = val - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 15:
                        break
            
            if not candidates and scan_limit < upper:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(c):
                        if rows[r][i] == -1:
                            continue
                        d = val - rows[r][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 15:
                            break
            
            return candidates

        def count_unfilled():
            count = 0
            for r in range(n_rows):
                for c in range(1, row_len):
                    if rows[r][c] == -1:
                        count += 1
            return count

        def solve():
            if time.time() > start + time_limit:
                return False
            
            if count_unfilled() == 0:
                return True
            
            best_pos = None
            best_candidates = None
            best_count = float('inf')
            
            for r in range(n_rows):
                for c in range(1, row_len):
                    if rows[r][c] == -1:
                        cands = get_candidates(r, c, target_scope)
                        if len(cands) == 0:
                            return False
                        if len(cands) < best_count:
                            best_count = len(cands)
                            best_pos = (r, c)
                            best_candidates = cands
                            if best_count == 1:
                                break
                if best_count == 1:
                    break
            
            if best_pos is None:
                return True
            
            r, c = best_pos
            random.shuffle(best_candidates)
            
            for val in best_candidates:
                rows[r][c] = val
                new_diffs = []
                for i in range(c):
                    if rows[r][i] != -1:
                        d = val - rows[r][i]
                        new_diffs.append((d, r, i))
                        add_diff(d, r, c)
                
                # Forward check
                forward_ok = True
                for r2 in range(n_rows):
                    for c2 in range(1, row_len):
                        if r2 == r and c2 == c:
                            continue
                        if rows[r2][c2] == -1:
                            prev_v = rows[r2][c2-1] if c2 > 0 else -1
                            low = prev_v + 1
                            high = target_scope
                            if low <= high:
                                found_any = False
                                for test_val in [low, (low+high)//2, high]:
                                    if test_val < low or test_val > high:
                                        continue
                                    valid = True
                                    for i in range(c2):
                                        if rows[r2][i] != -1:
                                            d = test_val - rows[r2][i]
                                            if d in used_diffs:
                                                valid = False
                                                break
                                    if valid:
                                        found_any = True
                                        break
                                if not found_any:
                                    forward_ok = False
                                    break
                    if not forward_ok:
                        break
                
                if forward_ok and solve():
                    return True
                
                rows[r][c] = -1
                for d, _, _ in new_diffs:
                    remove_diff(d, r, c)
            
            return False

        if solve():
            return [r[:] for r in rows]
        return None

    # Phase 1: Get a baseline with greedy
    best_rows = greedy_init(0.12, 25)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search Refinement
    local_time = 45.0
    elapsed = time.time() - start_time
    if elapsed > deadline - 55:
        local_time = max(10, deadline - elapsed - 25)
    
    if local_time > 5:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope
    
    # Phase 3: MRV Backtracking to try to beat the current best
    current_target = best_scope - 1
    seed_counter = 0
    
    while time.time() < deadline - 5:
        if current_target < 112:
            break
        
        search_time = 8.0
        remaining = deadline - time.time() - 3
        if remaining < search_time:
            search_time = max(1.0, remaining)
        
        if search_time < 1.0:
            break

        result = mrv_backtracking(n_rows, k, current_target, search_time, seed=seed_counter)
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
            if time.time() < deadline - 8:
                result_retry = mrv_backtracking(n_rows, k, current_target, 5.0, seed=seed_counter)
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
    
    # Phase 4: Final high-precision polish pass
    if time.time() < deadline - 2:
        polish_time = min(10.0, deadline - time.time() - 2)
        if polish_time > 2:
            polished_rows, polished_scope = local_search(best_rows, polish_time)
            if validate(polished_rows):
                best_rows = polished_rows
                best_scope = polished_scope
    
    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
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
