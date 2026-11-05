"""
Difference triangle set (7,5) solver.

Strategy:
1. Fast greedy construction to get an initial feasible solution.
2. CSP Backtracking search using bitmasks for difference tracking.
   - Builds rows sequentially.
   - For each element, selects candidates based on a heuristic cost (sum of 1/diff) 
     to balance the usage of the difference pool.
   - Uses randomized restarts to explore different parts of the search space.
   - Prunes based on a target scope.
3. Local search polishing to reduce the scope of the best found solution.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_diffs = []
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.append(r[j] - r[i])
    return len(all_diffs) == len(set(all_diffs))

def _compute_all_diffs(rows):
    """Return set of all differences used."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used

def _greedy_construct(n_rows, k, deadline):
    """
    Fast greedy construction.
    Picks the smallest valid next element for each row, column by column.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = 0  # Bitmask
    
    for pos in range(1, k + 1):
        # Process rows in order of their current last value to keep them balanced
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            x = prev_val + 1
            found = False
            while True:
                if time.time() > deadline:
                    return rows
                # Check validity of x
                conflict = False
                new_mask = 0
                for a in row:
                    d = x - a
                    # d must be positive and within reasonable bounds
                    # Max possible diff is bounded by current search scope estimate, 
                    # but for greedy we just check if bit is set.
                    # We assume x won't exceed 1000 in greedy.
                    if d > 1024: 
                        # Safety, though unlikely for k=5
                        break
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if not conflict:
                    row.append(x)
                    used_diffs |= new_mask
                    found = True
                    break
                x += 1
            
            if not found:
                # Should not happen for reasonable sizes, but handle gracefully
                break
    return rows

def _local_search(rows, k, deadline):
    """
    Local search to reduce scope.
    Tries to move elements to smaller values if possible.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            # Try to reduce elements from right to left (larger indices first)
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                # Try to decrease old_val
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    # Check if move is valid
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                            new_diffs.add(new_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                            new_diffs.add(row[i] - new_val)
                    
                    remaining = used - old_diffs
                    if new_diffs & remaining:
                        continue
                    
                    # Move is valid
                    row[pos] = new_val
                    used = (used - old_diffs) | new_diffs
                    improved = True
                    break
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    # Evaluator kills at 150s. Leave 10s margin.
    deadline = start_time + 140.0
    
    # Step 1: Fast greedy initial solution
    rows = _greedy_construct(n_rows, k, deadline - 10)
    if rows is None or len(rows) != n_rows or any(len(r) < k + 1 for r in rows):
        # Fallback: simple construction if greedy failed (unlikely)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                conflict = False
                new_mask = 0
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if not conflict:
                    row.append(x)
                    used |= new_mask
                x += 1
            rows.append(row)
            
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    
    # Step 2: Local search polish on the greedy solution
    if time.time() < deadline - 5:
        polished = _local_search(best_rows, k, deadline - 5)
        sc = _scope(polished)
        if sc < best_scope:
            best_scope = sc
            best_rows = polished
    
    # Step 3: CSP Backtracking Search
    # We try to find a solution with scope < best_scope.
    # We iterate on target_scope from a lower bound up to best_scope - 1.
    
    # Lower bound: Total diffs = n_rows * k * (k+1) / 2.
    # For 7, 5: 7 * 15 = 105 diffs. So max diff must be >= 105.
    lower_bound = n_rows * k * (k + 1) // 2
    if lower_bound < 105:
        lower_bound = 105
        
    # If we already found 105, we are optimal (counting bound).
    if best_scope <= lower_bound:
        return best_rows
    
    # Search for better solutions
    # We will try target scopes from best_scope - 1 down to lower_bound?
    # Or just run a search with a tight upper bound and let it find the minimum?
    # The inspiration suggests:
    # 1. Start with a high target (e.g. current best).
    # 2. Run search.
    # 3. If found, update best.
    # 4. Repeat with new best as target (looking for < best).
    
    current_target = best_scope
    seed_counter = 0
    max_attempts = 50 # Safety cap
    
    while time.time() < deadline - 2.0 and current_target > lower_bound and seed_counter < max_attempts:
        # Try to find a solution with scope < current_target
        # So the maximum allowed value is current_target - 1
        target_scope = current_target - 1
        
        found_solution = None
        rng = random.Random(seed_counter)
        
        # CSP State
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row, current_max_val):
            nonlocal used_diffs, rows, found_solution
            
            if time.time() > deadline - 2.0:
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < current_target: # Strictly better
                    found_solution = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0], current_max_val):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for next element x
            # x must be > last_val
            # x must be < current_target (strictly, because scope < current_target means max entry <= current_target - 1)
            upper_bound = current_target - 1
            
            # Generate candidates
            candidates = []
            
            # Optimization: 
            # If we are building the last row, we might be able to prune more aggressively.
            # For now, simple candidate generation.
            
            for x in range(last_val + 1, upper_bound + 1):
                # Check if new differences conflict with used_diffs
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    # d must be <= upper_bound (since max entry is upper_bound, max diff is upper_bound)
                    # Actually max diff is x - 0 = x <= upper_bound.
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Heuristic: Cost = sum(1/d). Lower cost is better (prefers larger diffs).
                # This helps in packing because it avoids using small diffs which are "scarcer" 
                # in terms of being easily blocked by other rows? 
                # Actually, small diffs are more likely to be in use. 
                # Using a large diff keeps the small diff pool intact for other rows that might need them.
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                # Add small random noise to break ties and allow exploration
                noise = rng.random() * 0.05
                
                candidates.append((cost + noise, x, new_mask))
                
                # Limit candidates to avoid explosion
                # If we have too many, we pick the best ones.
                # For k=5, the number of valid x is usually small due to constraints.
                if len(candidates) > 20:
                    break
            
            if not candidates:
                return False
            
            # Sort candidates by cost
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates
            # Randomize the order of the top candidates to explore different branches
            top_n = min(5, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                # Recurse
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x], max(current_max_val, x)):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        # Run the search
        _backtrack(0, [0], 0)
        
        if found_solution:
            sc = _scope(found_solution)
            if sc < best_scope:
                best_scope = sc
                best_rows = found_solution
                # Polish the new best
                if time.time() < deadline - 2.0:
                    polished = _local_search(best_rows, k, deadline - 2.0)
                    sc2 = _scope(polished)
                    if sc2 < best_scope:
                        best_scope = sc2
                        best_rows = polished
                
                # Update target to search for even better
                current_target = best_scope
                seed_counter += 1
            else:
                # Found a solution but not better than current best? 
                # This can happen if current_target was set to best_scope, 
                # and we found a solution with scope == current_target - 1 which is still > best_scope?
                # No, we check `if actual_scope < current_target`.
                # If we didn't find a solution, we just retry with different seed.
                seed_counter += 1
        else:
            # No solution found with this target.
            # If we failed to find ANY solution with scope < current_target, 
            # it might mean current_target is too tight or we got stuck.
            # We should probably relax the target? No, we want to minimize.
            # If we can't find < current_target, maybe current_target is already minimal?
            # Or we just need a different search path.
            seed_counter += 1
            # To avoid infinite loop on a very tight bound, we can break if we've tried many seeds
            # for the same target.
            if seed_counter % 5 == 0:
                # Try a slightly looser target? 
                # Actually, if we can't find < T, we might not find < T-1.
                # But maybe we can find a solution with scope T that is better structured?
                # We only care about the global min scope.
                # Let's just keep trying with different seeds for the same target a few times.
                pass
    
    return best_rows

# EVOLVE-BLOCK-END
