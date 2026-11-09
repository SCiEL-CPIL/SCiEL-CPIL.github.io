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
        """
        Generate a solution using randomized greedy.
        """
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider search for better diversity
                limit = x + 1000 + row_idx * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 50:
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
                    # Use weights that favor smaller but allow exploration
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

    def backtrack_min_row(forbidden_diffs, k, max_limit, seed=0):
        """
        Attempts to construct a row of size k+1 starting at 0, with max element <= max_limit,
        such that all differences are not in forbidden_diffs and are unique within the row.
        Returns the row with the minimal possible max element, or None if not found.
        Uses backtracking with pruning.
        """
        random.seed(seed)
        best_row = None
        best_max = max_limit + 1
        
        # We want to find a row with min max.
        # We can search for a valid row with max <= current_best_max.
        # We'll try to find any valid row, and keep the one with smallest max.
        
        # To make it faster, we can do a depth-first search.
        # State: (current_row, current_max, used_diffs_in_row)
        
        def dfs(current_row, current_max, used_in_row):
            nonlocal best_row, best_max
            
            if len(current_row) == k_plus_1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = list(current_row)
                return True
            
            # Pruning: if current_max is already >= best_max, we might still find a better one later?
            # No, because we are increasing current_max. If current_max >= best_max, 
            # any extension will have max >= current_max >= best_max. 
            # So we can prune if we are looking for strictly better.
            # However, if best_max is infinity initially, we just want any valid row.
            # Actually, we want the MINIMAL max. 
            # If we find a valid row with max M, we set best_max = M.
            # Then we can prune branches where current_max >= best_max.
            
            if current_max >= best_max:
                return False

            # Determine the range for the next element x
            # x must be > current_max
            # Also, the difference x - current_row[0] = x must be < best_max? No, x can be up to best_max.
            # But we want to find small x first to potentially keep max small?
            # Actually, the max of the row is the last element (since it's increasing).
            # So if we pick a small x now, we might be able to pick smaller elements later?
            # Not necessarily. But picking smaller x keeps the "running max" lower.
            
            # Heuristic: Try smaller x first.
            start_x = current_max + 1
            end_x = best_max - 1  # We don't need to try x >= best_max if we already have a solution with max best_max
            # If best_max is infinity, end_x can be large. But we have a global limit from the caller.
            # Let's use max_limit as a hard cap for x.
            if best_max == max_limit + 1:
                end_x = max_limit
            else:
                end_x = min(max_limit, best_max - 1)
                
            if start_x > end_x:
                return False

            # To allow some randomness and avoid getting stuck, we can shuffle the order of x slightly
            # but for minimal max, sequential is usually good. 
            # Let's stick to sequential for determinism in finding the minimal, 
            # but we call this function multiple times with different seeds in the outer loop?
            # Actually, the structure of the row matters. 
            # Let's try sequential first. If it finds a solution, it's the minimal one for that branch.
            
            for x in range(start_x, end_x + 1):
                new_diffs = {x - a for a in current_row}
                
                # Check conflicts with forbidden and internal
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                if len(new_diffs) != len(current_row):
                    # This check is redundant if the set logic above holds, but safe
                    continue
                
                # Recurse
                current_row.append(x)
                used_in_row.update(new_diffs)
                
                if dfs(current_row, x, used_in_row):
                    # If we found a solution, we can return early if we are just looking for ANY solution
                    # But we want the MINIMAL. 
                    # So we continue searching for a smaller max?
                    # Actually, if we return True, the caller will stop. 
                    # To find the minimal, we should NOT return True immediately unless we found a very good one.
                    # But this is exponential. 
                    # Let's just find the FIRST valid row with max <= max_limit, and hope it's good enough?
                    # Or we can keep searching for a smaller one within the same call?
                    # The current logic: if dfs returns True, it means a solution was found (and best_row updated).
                    # We should continue the loop to see if we can find a SMALLER max?
                    # No, because x is increasing. If we found a solution with current_max=x, 
                    # any subsequent x will be larger, so the max will be larger.
                    # So the FIRST solution found in this branch (with increasing x) is the minimal for this branch.
                    # But we are iterating x from small to large. So the first valid x leads to the smallest possible next element.
                    # This greedy DFS should find a good candidate.
                    return True
                
                current_row.pop()
                used_in_row.difference_update(new_diffs)
            
            return False
        
        dfs([0], 0, set())
        
        if best_row is not None:
            return best_row
        return None

    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry with multiple strategies."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        stall_count = 0
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            
            # Find top N largest elements to try shrinking
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:  # Skip the 0
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:5]:  # Try up to 5 largest
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                
                if current_val <= 0:
                    continue
                
                # Determine valid range
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                # Remove current differences from used
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                
                new_val_found = None
                # Try smallest valid replacement first, then some random ones
                candidates_to_try = list(range(min_val, min(current_val, min_val + 200)))
                # Add some random candidates in between
                for _ in range(20):
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
                    break  # Restart loop with updated elements
                else:
                    used |= old_diffs
            
            if not found_improvement:
                # Try a swap move: exchange two elements from different rows
                if stall_count < 3 and time.time() - start < max_time - 0.5:
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
                                    
                                    # Check if swapping maintains increasing order
                                    if i1 > 0 and v2 <= rows[r1][i1-1]:
                                        continue
                                    if i1 < len(rows[r1])-1 and v2 >= rows[r1][i1+1]:
                                        continue
                                    if i2 > 0 and v1 <= rows[r2][i2-1]:
                                        continue
                                    if i2 < len(rows[r2])-1 and v1 >= rows[r2][i2+1]:
                                        continue
                                    
                                    # Remove diffs
                                    old_diffs1 = set()
                                    for a in rows[r1]:
                                        if a != v1:
                                            old_diffs1.add(abs(a - v1))
                                    old_diffs2 = set()
                                    for a in rows[r2]:
                                        if a != v2:
                                            old_diffs2.add(abs(a - v2))
                                    used -= (old_diffs1 | old_diffs2)
                                    
                                    # Check new diffs
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
                
                if not found_improvement:
                    improved = False
                    stall_count += 1
                    if stall_count > 5:
                        break
        
        return best_rows

    def rebuild_row(rows, row_idx, max_limit, seed=0):
        """
        Rebuild a specific row using backtracking, given the constraints from other rows.
        """
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        used = set()
        for row in other_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        # Try to find a minimal row
        new_row = backtrack_min_row(used, k, max_limit, seed)
        if new_row is not None:
            return new_row
        return None

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Generate initial solutions with greedy
    num_seeds = 100
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 20:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            if best_score <= 112:
                break
        
        # Apply local search
        if remaining > 15:
            ls_time = min(5.0, remaining * 0.2)
            if ls_time > 1.0:
                polished = local_search(rows, max_time=ls_time)
                polished_score = get_score(polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_rows = [list(r) for r in polished]
                
                if best_score <= 112:
                    break
    
    # Phase 2: Intensive local search on the best solution
    if best_rows is not None:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 10:
            ls_time = min(remaining * 0.5, 30.0)
            if ls_time > 5.0:
                final_rows = local_search(best_rows, max_time=ls_time)
                final_score = get_score(final_rows)
                if final_score < best_score:
                    best_score = final_score
                    best_rows = [list(r) for r in final_rows]
                
                # Try rebuilding rows with backtracking if we have time
                if best_score > 115:
                    elapsed2 = time.time() - start_time
                    remaining2 = time_limit - elapsed2
                    if remaining2 > 10:
                        for r_idx in range(n_rows):
                            if time.time() - start_time > time_limit - 5:
                                break
                            # Try to rebuild this row with a lower max
                            current_max = max(best_rows[r_idx])
                            # Try to find a row with max < current_max
                            new_row = backtrack_min_row(compute_used([r for i, r in enumerate(best_rows) if i != r_idx]), k, current_max - 1, seed=r_idx)
                            if new_row is not None:
                                best_rows[r_idx] = new_row
                                new_score = get_score(best_rows)
                                if new_score < best_score:
                                    best_score = new_score
                                    if best_score <= 112:
                                        break

    if best_rows is None:
        # Fallback to simple greedy
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
    
    return best_rows
