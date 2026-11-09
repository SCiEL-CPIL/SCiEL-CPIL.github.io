import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard - use 115s to be safe for both 120s and 150s limits
    start_time = time.time()
    timeout = 115
    
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
    
    # 2. Improved Backtracking Search with better pruning
    def dfs_search(target_max, seed, max_nodes=50000):
        random.seed(seed)
        rows_found = []
        node_count = [0]
        
        def build_row(current_row, used_mask):
            if time.time() - start_time > timeout:
                return False
            
            if len(current_row) == k_plus_1:
                rows_found.append(list(current_row))
                return True
            
            # Pruning: Check if it's possible to complete the row
            slots_left = k_plus_1 - len(current_row)
            if current_row[-1] + slots_left > target_max:
                return False
            
            # Additional pruning: check if enough unused differences remain
            # Each new element adds len(current_row) new differences
            # Total differences needed = slots_left * (len(current_row) + 1) / 2 approximately
            # But we need to be more precise: for each new element at position p,
            # it creates p new differences with previous elements
            
            # Find valid candidates with adaptive branching
            # Use smaller branching factor when close to target
            remaining = target_max - current_row[-1]
            branching_factor = max(3, min(8, remaining // max(1, slots_left)))
            
            candidates = []
            x = current_row[-1] + 1
            
            while x <= target_max:
                valid = True
                new_diffs = 0
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if valid:
                    candidates.append((x, new_diffs))
                    if len(candidates) >= branching_factor:
                        break
                x += 1
            
            if not candidates:
                return False
            
            # Order candidates: prefer smaller values (greedy bias) but with some randomness
            # This helps find compact solutions faster
            candidates.sort(key=lambda c: c[0])
            
            # With some probability, shuffle to explore different branches
            if random.random() < 0.3:
                random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                node_count[0] += 1
                if node_count[0] > max_nodes or time.time() - start_time > timeout:
                    return False
                
                if build_row(current_row + [x], used_mask | new_diffs):
                    return True
            
            return False
        
        def build_all_rows(row_idx, used_mask):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if build_row([0], used_mask):
                # Calculate diffs for the row just added
                last_row = rows_found[-1]
                row_diffs = 0
                for i in range(len(last_row)):
                    for j in range(i + 1, len(last_row)):
                        d = last_row[j] - last_row[i]
                        row_diffs |= (1 << d)
                
                if build_all_rows(row_idx + 1, used_mask | row_diffs):
                    return True
            
            # Backtrack
            if rows_found:
                rows_found.pop()
            
            return False
        
        if build_all_rows(0, 0):
            return rows_found
        return None

    # Try to find better solutions
    current_target = best_max
    seeds_tried = 0
    max_seeds_per_target = 15
    
    while time.time() - start_time < timeout:
        current_target -= 1
        if current_target < 1:
            break
        
        found = False
        for seed in range(max_seeds_per_target):
            if time.time() - start_time > timeout:
                break
            
            res = dfs_search(current_target, seeds_tried)
            seeds_tried += 1
            
            if res:
                best_rows = res
                best_max = current_target
                found = True
                break
        
        if not found:
            break

    # 3. Enhanced Local Search Polish
    def local_search_polish(rows):
        rows = [list(r) for r in rows]
        used = compute_used_mask(rows)
        current_max = get_max(rows)
        improved = True
        stuck_count = 0
        
        while improved and time.time() - start_time < timeout:
            improved = False
            stuck_count += 1
            
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
                if stuck_count > 10:
                    break
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
                    stuck_count = 0
                    break
            
            if not found_replacement:
                # Try reducing other large elements if stuck
                if stuck_count > 5:
                    # Pick a random non-max element to try reducing
                    all_positions = []
                    for r in range(n_rows):
                        for i in range(len(rows[r])):
                            if not (r == max_r and i == max_i):
                                all_positions.append((r, i))
                    
                    if all_positions:
                        random.shuffle(all_positions)
                        for r, i in all_positions[:5]:  # Try up to 5 random positions
                            row2 = rows[r]
                            cv = row2[i]
                            mv = row2[i - 1] + 1 if i > 0 else 1
                            ma = cv - 1
                            if i < len(row2) - 1:
                                ma = min(ma, row2[i + 1] - 1)
                            
                            if ma < mv:
                                continue
                            
                            od = 0
                            for a in row2:
                                if a != cv:
                                    d = abs(a - cv)
                                    od |= (1 << d)
                            
                            uwo = used & ~od
                            
                            for v in range(mv, ma + 1):
                                valid = True
                                nd = 0
                                for a in row2:
                                    if a != cv:
                                        d = abs(a - v)
                                        if (uwo >> d) & 1:
                                            valid = False
                                            break
                                        if (nd >> d) & 1:
                                            valid = False
                                            break
                                        nd |= (1 << d)
                                
                                if valid:
                                    row2[i] = v
                                    used = uwo | nd
                                    improved = True
                                    found_replacement = True
                                    break
                            
                            if found_replacement:
                                break
            
            if not found_replacement and stuck_count > 20:
                break

        return rows, get_max(rows)

    polished_rows, polished_max = local_search_polish(best_rows)
    if polished_max < best_max:
        best_rows = polished_rows
        best_max = polished_max

    return best_rows
# EVOLVE-BLOCK-END
