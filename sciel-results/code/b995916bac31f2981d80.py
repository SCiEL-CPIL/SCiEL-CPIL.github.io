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
                    # Since we iterate x from small to large, the first solution found in this branch
                    # will have the smallest possible last element for this prefix.
                    # However, a different prefix might yield a smaller overall max.
                    # So we shouldn't return True immediately unless we are confident.
                    # Actually, if we find a solution with max M, and we are exploring x in increasing order,
                    # any other solution in this subtree will have max >= M.
                    # But other subtrees (different choices for previous elements) might have max < M.
                    # So we must continue searching other branches unless we prove optimality.
                    # For now, let's just record the best and continue.
                    pass
                
                current_row.pop()
                used_in_row.difference_update(new_diffs)
                
            return False

        dfs([0], 0, set())
        
        return best_row

    def local_search(rows, max_time=5.0):
        """
        Local search using backtracking row rebuilds and element shrinks.
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
            seed_counter += 1
            
            # Strategy 1: Shrink largest elements (fast local moves)
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:5]:
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
                # Try smallest valid replacement
                for v in range(min_val, min(current_val, min_val + 50)):
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
            
            # Strategy 2: Rebuild the row with the current global maximum using Backtracking
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            
            if max_r != -1 and time.time() - start < max_time - 1.0:
                # Compute forbidden diffs from other rows
                other_rows = [r for i, r in enumerate(rows) if i != max_r]
                forbidden = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            forbidden.add(r[j] - r[i])
                
                # Try to find a better row
                # Limit: we want to improve the global max, so limit is max_val - 1
                # But also try to improve the row's own max even if it's not the global max?
                # Let's focus on the global max first.
                limit = max_val - 1
                
                # Add some randomness to the backtracking? 
                # The backtracking function is deterministic for a given seed? 
                # Actually, my backtracking function above is deterministic (sequential x).
                # To explore different structures, we might need to vary the search order.
                # For now, let's just run it. If it finds a row with max < max_val, it's a win.
                
                new_row = backtrack_min_row(forbidden, k, limit, seed=seed_counter)
                
                if new_row is not None:
                    new_max = max(new_row)
                    if new_max < max_val:
                        # Update rows
                        old_row = rows[max_r]
                        old_diffs = set()
                        for i in range(len(old_row)):
                            for j in range(i+1, len(old_row)):
                                old_diffs.add(old_row[j] - old_row[i])
                        
                        new_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                new_diffs.add(new_row[j] - new_row[i])
                        
                        used -= old_diffs
                        used |= new_diffs
                        rows[max_r] = new_row
                        
                        score = get_score(rows)
                        if score < best_score:
                            best_score = score
                            best_rows = [list(r) for r in rows]
                        improved = True
                        continue

            # Strategy 3: If no improvement, try rebuilding a random row with a slightly higher limit
            # to see if we can restructure it to allow other rows to shrink later?
            # This is complex. Let's stick to stall count.
            if stall_count > 10:
                break

        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    num_seeds = 50
    
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 10:
            break
        
        # Allocate time for local search
        if seed < 10:
            ls_time = min(2.0, remaining * 0.1)
        elif seed < 30:
            ls_time = min(5.0, remaining * 0.15)
        else:
            ls_time = min(10.0, remaining * 0.2)
            
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
        if best_score <= 130 and remaining > 20 and seed > 10:
            final_rows = local_search(best_rows, max_time=min(remaining * 0.3, 15.0))
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
    
    # Final polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.8, 10.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            
    return best_rows
