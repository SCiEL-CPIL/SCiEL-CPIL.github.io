import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
        
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def backtrack_row(row, used, k_plus_1, max_depth=8, time_limit=None):
        """
        Backtracking search to extend a partial row to k_plus_1 elements.
        Returns a complete row or None if no solution found within limits.
        """
        if time_limit is not None and time.time() > time_limit:
            return None
        if len(row) == k_plus_1:
            return row[:]
        
        current_max = row[-1]
        new_used = set(used)
        
        # Try candidates in order of preference (smaller first)
        x = current_max + 1
        limit = x + 200  # Search window
        attempts = 0
        
        while x <= limit and attempts < 50:
            new_diffs = {x - a for a in row}
            # Check for collisions within new_diffs
            if len(new_diffs) == len(row):
                if not (new_diffs & new_used):
                    row.append(x)
                    new_used.update(new_diffs)
                    result = backtrack_row(row, new_used, k_plus_1, max_depth, time_limit)
                    if result is not None:
                        return result
                    row.pop()
                    new_used -= new_diffs
            x += 1
            attempts += 1
        
        return None

    def greedy_solution(seed=0, fixed_used=None, target_row_idx=None):
        """
        Generate a solution using randomized greedy with backtracking fallback.
        If fixed_used is provided, it treats those differences as already taken
        and only generates the row at target_row_idx (or all if None, but usually
        used for rebuilds of a single row).
        """
        random.seed(seed)
        used = set()
        if fixed_used is not None:
            used.update(fixed_used)
            
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            success = False
            
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                limit = x + 1000 + row_idx * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 50:
                            break
                    x += 1
                
                if not candidates:
                    # Backtracking fallback
                    result = backtrack_row(row, used, k_plus_1, time_limit=time.time() + 0.5)
                    if result is not None:
                        row = result
                        break
                    else:
                        # Fallback: just find any valid next element
                        x = current_max + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if len(new_diffs) == len(row) and not (new_diffs & used):
                                candidates.append(x)
                                break
                            x += 1
                
                # Weighted random selection: prefer smaller values but with diversity
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    alpha = 0.005
                    weights = [1.0 / (1.0 + c * alpha) for c in candidates[:20]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:20], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows

    def rebuild_row(rows, row_idx, seed, max_time=2.0):
        """
        Attempts to rebuild a specific row to reduce the global max using backtracking.
        Returns the new rows if an improvement is found, else None.
        """
        start = time.time()
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        
        # Compute differences used by other rows
        used_by_others = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_by_others.add(r[j] - r[i])
        
        random.seed(seed)
        
        best_new_row = None
        best_new_max = max(rows[row_idx])
        
        # Try multiple approaches: greedy first, then backtracking with different starting points
        for attempt in range(15):
            if time.time() - start > max_time:
                break
            
            row = [0]
            new_row_used = set()
            current_max = 0
            valid = True
            
            # Try different strategies based on attempt number
            if attempt < 5:
                # Pure greedy
                while len(row) < k_plus_1:
                    x = current_max + 1
                    found = False
                    limit = x + 500
                    while x <= limit:
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used_by_others) and not (new_diffs & new_row_used):
                            row.append(x)
                            new_row_used.update(new_diffs)
                            current_max = x
                            found = True
                            break
                        x += 1
                    if not found:
                        valid = False
                        break
            else:
                # Backtracking-based construction
                result = backtrack_row(row, used_by_others, k_plus_1, time_limit=time.time() + 0.3)
                if result is None:
                    valid = False
                else:
                    row = result
                    valid = True
            
            if valid:
                row_max = max(row)
                if row_max < best_new_max:
                    best_new_max = row_max
                    best_new_row = row
        
        if best_new_row is not None:
            new_rows = [list(r) for r in rows]
            new_rows[row_idx] = best_new_row
            return new_rows
            
        return None

    def local_search(rows, max_time=5.0):
        """
        Hybrid local search: Shrink elements, Rebuild rows with backtracking, Swap elements.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = compute_used(rows)
        
        improved = True
        stall_count = 0
        seed_counter = 0
        
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            
            # Strategy 1: Shrink largest elements (more aggressive search range)
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:8]:
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                if current_val <= 0:
                    continue
                
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                
                new_val_found = None
                # Try wider range for replacement
                candidates_to_try = list(range(min_val, min(current_val, min_val + 200)))
                
                for v in candidates_to_try:
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid and len(new_diffs) == len(row) - 1:
                        new_val_found = v
                        break
                
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    used |= old_diffs
            
            if found_improvement:
                improved = True
                continue
            
            # Strategy 2: Rebuild the row with the current global maximum (with backtracking)
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            
            if max_r != -1 and time.time() - start < max_time - 0.5:
                seed_counter += 1
                new_rows = rebuild_row(rows, max_r, seed_counter, max_time=1.5)
                if new_rows:
                    score = get_score(new_rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                        used = compute_used(rows)
                        improved = True
                        continue

            # Strategy 3: Try rebuilding multiple rows, not just the max one
            if stall_count >= 2 and time.time() - start < max_time - 0.5:
                # Sort rows by max value, try rebuilding the top 2
                row_order = sorted(range(n_rows), key=lambda r: max(rows[r]), reverse=True)
                for r_to_rebuild in row_order[:2]:
                    if time.time() - start > max_time - 0.3:
                        break
                    seed_counter += 1
                    new_rows = rebuild_row(rows, r_to_rebuild, seed_counter, max_time=1.0)
                    if new_rows:
                        score = get_score(new_rows)
                        if score < best_score:
                            best_score = score
                            best_rows = [list(r) for r in new_rows]
                            rows = new_rows
                            used = compute_used(rows)
                            improved = True
                            break
            
            # Strategy 4: Swap (less frequent)
            if stall_count < 4 and time.time() - start < max_time - 0.5:
                swap_done = False
                for r1 in range(n_rows):
                    for r2 in range(r1 + 1, n_rows):
                        if time.time() - start > max_time - 0.2:
                            break
                        for i1 in range(1, len(rows[r1])):
                            for i2 in range(1, len(rows[r2])):
                                v1 = rows[r1][i1]
                                v2 = rows[r2][i2]
                                if v1 == v2:
                                    continue
                                
                                if i1 > 0 and v2 <= rows[r1][i1-1]:
                                    continue
                                if i1 < len(rows[r1])-1 and v2 >= rows[r1][i1+1]:
                                    continue
                                if i2 > 0 and v1 <= rows[r2][i2-1]:
                                    continue
                                if i2 < len(rows[r2])-1 and v1 >= rows[r2][i2+1]:
                                    continue
                                    
                                old_diffs1 = set()
                                for a in rows[r1]:
                                    if a != v1:
                                        old_diffs1.add(abs(a - v1))
                                old_diffs2 = set()
                                for a in rows[r2]:
                                    if a != v2:
                                        old_diffs2.add(abs(a - v2))
                                used -= (old_diffs1 | old_diffs2)
                                
                                new_diffs1 = set()
                                valid = True
                                for a in rows[r1]:
                                    if a != v1:
                                        d = abs(a - v2)
                                        if d in used:
                                            valid = False
                                            break
                                        new_diffs1.add(d)
                                if valid:
                                    new_diffs2 = set()
                                    for a in rows[r2]:
                                        if a != v2:
                                            d = abs(a - v1)
                                            if d in used:
                                                valid = False
                                                break
                                            new_diffs2.add(d)
                                
                                if valid:
                                    rows[r1][i1] = v2
                                    rows[r2][i2] = v1
                                    used |= (new_diffs1 | new_diffs2)
                                    swap_done = True
                                    found_improvement = True
                                    score = get_score(rows)
                                    if score < best_score:
                                        best_score = score
                                        best_rows = [list(r) for r in rows]
                                    break
                                else:
                                    used |= (old_diffs1 | old_diffs2)
                            if swap_done:
                                break
                        if swap_done:
                            break
                    if swap_done:
                        break
                
                if swap_done:
                    improved = True
                    continue

            if not found_improvement:
                improved = False
                stall_count += 1
                if stall_count > 6:
                    break
        
        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds with adaptive time allocation
    num_seeds = 80  # Slightly fewer seeds to allow more time per seed
    
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 15:
            break
        
        # Allocate time for local search based on progress
        # Give more time to later seeds when we're making progress
        if best_score == float('inf'):
            ls_time = min(3.0, remaining * 0.15)
        elif best_score <= 140:
            ls_time = min(6.0, remaining * 0.2)
        elif best_score <= 125:
            ls_time = min(10.0, remaining * 0.3)
        else:
            ls_time = min(8.0, remaining * 0.25)
            
        if ls_time < 0.5:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search
        rows = local_search(rows, max_time=ls_time)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            if score <= 112:
                break
        
        # Intensive polish if we are close to the bound
        if best_score <= 135 and remaining > 25 and seed > 20:
            polish_time = min(remaining * 0.4, 20.0)
            if polish_time > 3:
                final_rows = local_search(best_rows, max_time=polish_time)
                final_score = get_score(final_rows)
                if final_score < best_score:
                    best_score = final_score
                    best_rows = [list(r) for r in final_rows]

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if len(new) == len(row) and not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 8:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.85, 15.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            
    return best_rows
