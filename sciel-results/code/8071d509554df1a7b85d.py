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

    def greedy_solution(seed=0):
        """Generate a solution using randomized greedy with adaptive candidate selection."""
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates with a wider search
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider for later elements
                limit = x + 800 + row_idx * 100
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 30:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Weighted random selection: prefer smaller values but with diversity
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    # Use exponential weighting to favor smaller candidates
                    weights = [1.0 / (1.0 + c * 0.01) for c in candidates[:10]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:10], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows

    def rebuild_row(rows, row_idx, seed, time_limit):
        """
        Attempts to rebuild a specific row to reduce the global max.
        Returns the new row if an improvement is found, else None.
        """
        # Compute differences used by other rows
        used_by_others = set()
        for i, r in enumerate(rows):
            if i == row_idx:
                continue
            for a in range(len(r)):
                for b in range(a+1, len(r)):
                    used_by_others.add(r[b] - r[a])
        
        current_row_max = max(rows[row_idx])
        # We want to find a row with max < current_row_max
        # If we find one, it's an improvement for that row's contribution to the global max.
        # Note: It might not reduce the *global* max if another row is larger,
        # but it frees up differences and might allow other rows to shrink later.
        # However, for the immediate score, only reducing the global max matters.
        # So we should target the row that *is* the global max, or any row that can be reduced.
        # Let's just try to find a valid row with max < current_row_max.
        
        random.seed(seed)
        
        best_new_row = None
        best_new_max = current_row_max
        
        # Backtracking function
        path = [0]
        internal_diffs = set()
        
        def bt(idx):
            nonlocal best_new_row, best_new_max
            if time.time() > time_limit:
                return False
            
            if idx == k_plus_1:
                if path[-1] < best_new_max:
                    best_new_max = path[-1]
                    best_new_row = list(path)
                return True
            
            last = path[-1]
            # Pruning: remaining elements must fit
            if best_new_max - 1 - last < (k_plus_1 - idx):
                return False
            
            # Collect valid candidates
            candidates = []
            # We search up to best_new_max - 1
            for x in range(last + 1, best_new_max):
                new_diffs = {x - a for a in path}
                if new_diffs & used_by_others:
                    continue
                if new_diffs & internal_diffs:
                    continue
                # Check internal uniqueness
                if len(new_diffs) != len(path):
                    continue
                candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Randomize order to escape local minima
            random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                path.append(x)
                internal_diffs.update(new_diffs)
                
                if bt(idx + 1):
                    return True
                
                path.pop()
                internal_diffs.difference_update(new_diffs)
            
            return False
        
        # Try a few random restarts for the rebuild
        for _ in range(3):
            path = [0]
            internal_diffs = set()
            # Reset best for this attempt? No, keep global best for this row rebuild call.
            # But if we find one, we can stop.
            if bt(1):
                break
        
        if best_new_row:
            return best_new_row
        return None

    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry with multiple strategies."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = compute_used(rows)
        
        improved = True
        stall_count = 0
        iter_count = 0
        
        while improved and time.time() - start < max_time:
            improved = False
            iter_count += 1
            
            # 1. Element Moves
            # Find top N largest elements to try shrinking
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
                candidates_to_try = list(range(min_val, min(current_val, min_val + 400)))
                for _ in range(40):
                    candidates_to_try.append(random.randint(min_val, current_val - 1))
                
                for v in candidates_to_try:
                    if v <= min_val - 1 or v >= current_val:
                        continue
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid:
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
            
            # 2. Swaps
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
                                else:
                                    used |= (old_diffs1 | old_diffs2)
                            if swap_done:
                                break
                        if swap_done:
                            break
                    if swap_done:
                        break
                
                if found_improvement:
                    improved = True
                    continue

            # 3. Row Rebuild
            # Try every 5 iterations or if stalled
            if iter_count % 5 == 0 or (not improved and time.time() - start < max_time - 1.0):
                # Identify the row with the largest max
                max_row_idx = 0
                max_val = -1
                for i, r in enumerate(rows):
                    if max(r) > max_val:
                        max_val = max(r)
                        max_row_idx = i
                
                # Attempt rebuild
                rebuild_seed = int(time.time() * 1000) % 100000
                new_row = rebuild_row(rows, max_row_idx, rebuild_seed, time_limit=0.5)
                
                if new_row is not None:
                    # Verify validity of new row in context
                    # (rebuild_row already ensures it, but double check score)
                    old_rows = [list(r) for r in rows]
                    rows[max_row_idx] = new_row
                    new_score = get_score(rows)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        used = compute_used(rows)
                        improved = True
                    else:
                        # Revert if no improvement in score (though it might help later, 
                        # for now we stick to score-improving moves to be safe)
                        rows = old_rows
                        # used is unchanged
            
            if not improved:
                stall_count += 1
                if stall_count > 10:
                    break
        
        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    # Reduce number of seeds to allow more time for local search + rebuilds
    num_seeds = 80 
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 15:
            break
        
        # Allocate time for local search
        # Later seeds get more time? Or consistent time?
        # Let's give each seed ~1.5s for local search.
        ls_time = min(1.5, remaining * 0.2)
        
        if ls_time < 0.5:
            break
            
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search with rebuilds
        rows = local_search(rows, max_time=ls_time)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            # Early exit if we hit the theoretical bound
            if score <= 112:
                break
        
        # If we've found a good solution, do a final intensive polish
        if best_score <= 130 and remaining > 30 and seed > 20:
            # Do one more intensive local search on the best solution
            # Increase time for this final push
            final_rows = local_search(best_rows, max_time=min(remaining * 0.4, 20.0))
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
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final validation and polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10 and best_score > 112:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.8, 25.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
    
    return best_rows
