import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit (evaluator kills at 150s, but prompt says 120s limit in one place, 150s in another. 
                   # "Each evaluation must finish within 120 s" in task contract. 
                   # "evaluator KILLS the program after 150s" in directive. 
                   # I will use 115s to be safe for both interpretations, leaving buffer for Python overhead.)
    
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
    # We use a greedy approach that picks the smallest valid number.
    # This provides a fast, valid baseline.
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
    
    # 2. Backtracking Search to improve
    # We try to find a solution with max < best_max.
    # We use a randomized DFS that explores a limited number of candidates at each step.
    
    def dfs_search(target_max, seed):
        random.seed(seed)
        used_global = 0
        rows_found = []
        
        def build_row(row_idx, current_row, used_mask):
            if time.time() - start_time > timeout:
                return False
            
            if len(current_row) == k_plus_1:
                rows_found.append(current_row)
                return True
            
            # Pruning: Check if it's possible to complete the row
            slots_left = k_plus_1 - len(current_row)
            if current_row[-1] + slots_left > target_max:
                return False
            
            # Find valid candidates
            # We scan from current_row[-1] + 1 upwards.
            # We collect up to 'branching_factor' valid candidates to keep search tree manageable.
            branching_factor = 5
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
            
            # Shuffle candidates to explore different branches
            random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                if build_row(row_idx, current_row + [x], used_mask | new_diffs):
                    return True
            
            return False
        
        def build_all_rows(row_idx, used_mask):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if build_row(row_idx, [0], used_mask):
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
    # We iterate downwards from best_max - 1
    current_target = best_max
    seeds_tried = 0
    max_seeds_per_target = 10
    
    while time.time() - start_time < timeout:
        current_target -= 1
        if current_target < 1:
            break
        
        found = False
        for seed in range(max_seeds_per_target):
            if time.time() - start_time > timeout:
                break
            
            # Use a unique seed for each attempt
            res = dfs_search(current_target, seeds_tried)
            seeds_tried += 1
            
            if res:
                best_rows = res
                best_max = current_target
                found = True
                break
        
        if not found:
            # If we failed to find a solution for this target, 
            # it's likely we won't find one for lower targets easily without more time.
            # However, sometimes a specific structure only works for a specific max.
            # We continue to lower targets but with fewer seeds? 
            # Or just break? 
            # Given the "decreasing target" strategy, if we can't fit in T, we can't fit in T-1 
            # if the constraints are monotonic? 
            # Actually, the set of valid configurations for max <= T is a subset of max <= T+1.
            # So if no solution exists for max <= T, no solution exists for max <= T-1.
            # Wait, is that true?
            # If a solution exists with max M, it is a valid solution for any target >= M.
            # If no solution exists for target T, it means no valid DTS has max <= T.
            # Therefore, no valid DTS has max <= T-1.
            # So we can break immediately if we fail to find a solution for current_target.
            break

    # 3. Local Search Polish
    # Try to reduce the max value by swapping elements.
    # This is a refinement step.
    
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
            # We want the smallest v that works, to minimize max.
            # Iterate from min_val to max_allowed
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
                    # Check if new_diffs are all distinct and not in used_without_old
                    # The loop above checks against used_without_old and new_diffs itself.
                    # We also need to ensure the count is correct? 
                    # len(row)-1 differences.
                    # Since we checked each d against new_diffs, they are distinct.
                    
                    # Apply change
                    row[max_i] = v
                    used = used_without_old | new_diffs
                    improved = True
                    found_replacement = True
                    break
            
            # If we didn't find a replacement for the max, 
            # we might try to reduce the second max? 
            # For simplicity, we only target the global max in this pass.
            # If stuck, we break to avoid infinite loop.
            if not found_replacement:
                # Try a random element? 
                # Or just break.
                break

        return rows, get_max(rows)

    polished_rows, polished_max = local_search_polish(best_rows)
    if polished_max < best_max:
        best_rows = polished_rows
        best_max = polished_max

    return best_rows
# EVOLVE-BLOCK-END
