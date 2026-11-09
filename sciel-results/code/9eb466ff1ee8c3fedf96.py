import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    k_plus_1 = k + 1
    
    # Bitmask helper: compute used differences mask
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

    # 1. Greedy initial solution - fast baseline
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
    
    # 2. Backtracking with forward checking and randomized restarts
    # Key improvement: use a priority-based candidate ordering that balances
    # small values with "room" for remaining elements
    
    def backtracking_search(target_max, seed):
        random.seed(seed)
        
        # Pre-compute: for each possible row length, what's the minimum
        # number we need to leave room?
        # If we have slots_left positions to fill, and current last is L,
        # we need L + slots_left <= target_max
        
        def build_row(current_row, used_mask, row_idx):
            if time.time() - start_time > timeout:
                return None
            
            if len(current_row) == k_plus_1:
                return current_row
            
            slots_left = k_plus_1 - len(current_row)
            # Pruning: can we still fit?
            if current_row[-1] + slots_left > target_max:
                return None
            
            # Find all valid candidates
            candidates = []
            x = current_row[-1] + 1
            
            # Limit candidate scan for efficiency
            scan_limit = min(target_max, current_row[-1] + 50)  # Look ahead at most 50
            
            while x <= scan_limit:
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
                x += 1
            
            if not candidates:
                # Try a wider scan
                x = scan_limit + 1
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
                    x += 1
            
            if not candidates:
                return None
            
            # Score candidates: prefer small, but also prefer those that
            # don't create large gaps (which waste difference space)
            def score_candidate(x, new_diffs):
                # Prefer smaller x
                base = x
                # Penalize if this creates a very large difference
                max_d = max(row[j] - row[i] for i in range(len(current_row)) for j in range(i+1, len(current_row) + 1) if True)
                # Actually just use x as primary heuristic
                return x
            
            # Shuffle for diversity, but bias towards small
            # Use a weighted selection: pick from top candidates with probability
            random.shuffle(candidates)
            
            # Try candidates in order, but with some randomization
            # Sort by a mix of value and randomness
            scored = [(x, nd, random.random() * 10 + x) for x, nd in candidates]
            scored.sort(key=lambda t: t[2])
            
            for x, new_diffs, _ in scored:
                result = build_row(current_row + [x], used_mask | new_diffs, row_idx)
                if result is not None:
                    return result
            
            return None
        
        def build_all_rows(row_idx, used_mask):
            if time.time() - start_time > timeout:
                return None
            
            if row_idx == n_rows:
                return []
            
            row = build_row([0], used_mask, row_idx)
            if row is None:
                return None
            
            # Compute differences for this row
            row_diffs = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    row_diffs |= (1 << d)
            
            rest = build_all_rows(row_idx + 1, used_mask | row_diffs)
            if rest is not None:
                return [row] + rest
            
            return None
        
        return build_all_rows(0, 0)

    # 3. Local search refinement
    def local_search_polish(rows):
        rows = [list(r) for r in rows]
        used = compute_used_mask(rows)
        current_max = get_max(rows)
        
        # Multiple passes to reduce max
        for _ in range(50):  # Limit iterations
            if time.time() - start_time > timeout:
                break
            
            improved = False
            # Find global maximum element
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
                # Try random element swaps
                random.seed(int(time.time() * 1000) % 10000)
                for _ in range(100):
                    if time.time() - start_time > timeout:
                        break
                    r = random.randint(0, n_rows - 1)
                    i = random.randint(1, len(rows[r]) - 1)  # Not the first element
                    row = rows[r]
                    current_val = row[i]
                    
                    min_val = row[i - 1] + 1
                    max_allowed = current_val - 1
                    if i < len(row) - 1:
                        max_allowed = min(max_allowed, row[i + 1] - 1)
                    
                    if max_allowed < min_val:
                        continue
                    
                    old_diffs = 0
                    for a in row:
                        if a != current_val:
                            d = abs(a - current_val)
                            old_diffs |= (1 << d)
                    
                    used_without_old = used & ~old_diffs
                    
                    # Try a random smaller value
                    if max_allowed >= min_val:
                        v = random.randint(min_val, max_allowed)
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
                        
                        if valid and v < current_val:
                            row[i] = v
                            used = used_without_old | new_diffs
                            improved = True
                            break
            
            if not improved:
                break
        
        return rows, get_max(rows)

    # 4. Main search: try to find solutions with decreasing max
    # Start from best_max and try to go lower
    
    # First, try a few random restarts at the current best_max to see if we can find
    # a different solution that local search can improve more
    for seed in range(20):
        if time.time() - start_time > timeout:
            break
        res = backtracking_search(best_max, seed)
        if res:
            polished, pmax = local_search_polish(res)
            if pmax < best_max:
                best_rows = polished
                best_max = pmax
    
    # Now try to find solutions with lower max
    current_target = best_max - 1
    seeds_per_target = 15
    
    while time.time() - start_time < timeout and current_target >= 1:
        found = False
        for seed in range(seeds_per_target):
            if time.time() - start_time > timeout:
                break
            
            res = backtracking_search(current_target, seed)
            if res:
                polished, pmax = local_search_polish(res)
                if pmax <= current_target:
                    best_rows = polished
                    best_max = pmax
                    found = True
                    break
        
        if not found:
            # If we can't find a solution for this target, try a few more seeds
            # but with a time limit
            extra_time = 5  # seconds
            extra_start = time.time()
            for seed in range(50, 100):
                if time.time() - extra_start > extra_time:
                    break
                if time.time() - start_time > timeout:
                    break
                res = backtracking_search(current_target, seed)
                if res:
                    polished, pmax = local_search_polish(res)
                    if pmax <= current_target:
                        best_rows = polished
                        best_max = pmax
                        found = True
                        break
        
        if not found:
            # Can't find a solution for this target, so we can't find one for lower targets
            # (since the constraint is monotonic: if no solution exists with max <= T,
            #  no solution exists with max <= T-1)
            break
        
        current_target = best_max - 1
    
    # Final polish
    polished, pmax = local_search_polish(best_rows)
    if pmax < best_max:
        best_rows = polished
        best_max = pmax
    
    return best_rows
# EVOLVE-BLOCK-END
