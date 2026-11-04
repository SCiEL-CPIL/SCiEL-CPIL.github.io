"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Theoretical lower bound: differences are distinct integers in [1, M]
    # So M >= total_diffs. However, differences are bounded by max_entry.
    # Actually, max_entry must be at least total_diffs is not strictly true for the value,
    # but the differences must fit in [1, max_entry].
    # So max_entry >= total_diffs.
    lower_bound = total_diffs

    # We want to find a set of n_rows rows, each of length k+1, starting with 0,
    # such that all differences a_j - a_i (i < j) are distinct across all rows.
    # Let M be the maximum entry. The differences are in [1, M].
    # We need total_diffs distinct differences, so M >= total_diffs.

    # Strategy:
    # 1. Generate a pool of candidate rows.
    #    A candidate row is a tuple (row_values, diff_mask).
    #    diff_mask is a bitmask where bit d is set if difference d is present.
    # 2. Use a backtracking search to select n_rows candidates from the pool
    #    such that their diff_masks are pairwise disjoint (AND is 0).
    # 3. To keep the pool manageable, we generate rows up to a certain max_value.
    #    We start with a max_value close to the lower bound and increase if needed.

    # Since the problem size for (4, 7) is small (total_diffs = 4 * 28 = 112),
    # M is likely around 112-134.
    # The number of possible rows of length 8 with max value M is C(M, 7).
    # For M=134, C(134, 7) is huge (~10^11). We cannot generate all.
    # We need to generate "good" rows or use a heuristic generation.
    
    # Alternative Strategy: Randomized Construction with Global Check
    # Generate random rows, check if they fit with current selection.
    # This is similar to the current approach but with a better heuristic for row generation.
    
    # Let's try a hybrid:
    # 1. Pre-compute a set of "valid" differences for a single row? No, row validity depends on internal diffs.
    # 2. Use a SAT-like approach? Too complex to implement quickly.
    
    # Let's refine the "Exact Cover" idea with a limited pool.
    # We will generate rows greedily but with randomization, storing them in a pool.
    # Then we select from the pool.
    
    # Actually, for small n_rows (4) and k (7), we can use a more aggressive search.
    # Let's try a different angle: **Simulated Annealing** or **Local Search** on the set of rows.
    # But the constraint is hard (distinct diffs).
    
    # Let's stick to the **Global Backtracking with Bitmasks** but improve the row generation.
    # The current program builds rows one by one. If it gets stuck, it restarts.
    # The problem is that early rows might block later rows.
    # A "lookahead" approach: when choosing a row, check if it leaves enough "space" for remaining rows.
    
    # Heuristic: A row "uses" a set of differences. The "cost" of a row is the number of small differences it uses, 
    # because small differences are more likely to conflict.
    # We want to distribute the differences.
    
    # New Strategy: **Iterative Deepening A* (IDA*)** or **Branch and Bound** on the maximum entry M.
    # We binary search on M? No, M is an integer.
    # We try to construct a solution with max entry <= M.
    # Start M at lower_bound, increase until success.
    
    # To make the search efficient:
    # 1. Represent used differences as a bitmask (Python int).
    # 2. When building a row, use backtracking.
    # 3. Key improvement: **Ordering of rows**. 
    #    Instead of building row 0, then row 1, etc., we can interleave? No, rows are independent sets.
    #    But we can sort the rows by their "difficulty" or "size".
    
    # Let's implement a robust backtracker that:
    # 1. Takes a target M.
    # 2. Tries to build n_rows rows with max entry <= M.
    # 3. Uses a bitmask for global difference tracking.
    # 4. Uses a heuristic to choose the next row's entries: prefer entries that create differences 
    #    that are "less used" or "larger" to save small differences for others?
    #    Actually, small differences are the bottleneck. We should try to use small differences 
    #    in rows that have fewer constraints?
    
    # Let's try a simple improvement first: **Better Candidate Ordering**.
    # In the current code, candidates are shuffled slightly.
    # Let's try sorting candidates by the number of conflicts they introduce? 
    # No, we check conflicts before adding.
    
    # Let's try **Random Restart with Decaying Temperature**.
    
    # Implementation Plan:
    # 1. Function `can_build_row(used_mask, current_row, target_len, M, rng)`:
    #    - Recursively tries to extend `current_row`.
    #    - For each candidate x, computes new_diffs_mask.
    #    - If (used_mask & new_diffs_mask) == 0, recurse.
    #    - Heuristic: Try x in an order that minimizes the "impact" on future rows.
    #      Impact metric: sum of (1/d) for d in new_diffs? Or just count of small diffs?
    #      Let's try: prefer x that creates larger differences.
    
    def try_construct_with_limit(M, seed):
        """
        Attempts to construct n_rows rows with max entry <= M.
        Returns list of rows or None.
        """
        rng = random.Random(seed)
        used_mask = 0
        rows = []
        
        # Precompute valid next steps? No, dynamic.
        
        def build_row(idx):
            if time.time() > deadline:
                return False
            
            if idx == n_rows:
                return True
            
            row = [0]
            
            def backtrack(pos, current_max):
                if time.time() > deadline:
                    return False
                
                if pos == k + 1:
                    rows.append(row[:])
                    return True
                
                # Candidates: current_max + 1 to M
                # We want to try values that leave "good" differences for others.
                # Let's generate candidates and sort them by a heuristic.
                
                start = current_max + 1
                end = M + 1
                
                candidates = []
                for x in range(start, end):
                    # Check if x is valid against current row and global used_mask
                    # Compute diffs with existing row elements
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        # Check if d is in used_mask
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        # Check if d is in new_diffs (internal conflict)
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        # Heuristic score: 
                        # We want to use differences that are "large" to preserve small ones?
                        # Or we want to use differences that are "unique" in some way?
                        # Let's try: score = -sum(1/d for d in new_diffs)
                        # Higher score (more negative) means we prefer smaller diffs? No.
                        # We want to AVOID small diffs if possible, so prefer large diffs.
                        # Score = sum(1/d). We want to MINIMIZE this score? 
                        # If we minimize sum(1/d), we prefer large d.
                        score = sum(1.0/d for d in new_diffs)
                        candidates.append((score, x, new_diffs))
                
                if not candidates:
                    return False
                
                # Sort candidates by score (ascending, so large diffs first)
                # Add some randomness to break ties or explore variety
                rng.shuffle(candidates) # Break ties randomly
                candidates.sort(key=lambda t: t[0])
                
                for score, x, new_diffs in candidates:
                    # Apply
                    row.append(x)
                    new_mask = 0
                    for d in new_diffs:
                        new_mask |= (1 << d)
                    
                    if backtrack(pos + 1, x):
                        return True
                    
                    # Undo
                    row.pop()
                
                return False
            
            if backtrack(1, 0):
                # Row successfully built and appended to `rows`
                # Update global used_mask
                # Wait, `rows` is modified inside `backtrack` which is inside `build_row`.
                # We need to update `used_mask` in the outer scope.
                # This closure approach is tricky with `used_mask`.
                # Let's pass used_mask as a mutable object or return it.
                # Simpler: make used_mask a list [0] to allow mutation in closure.
                return True
            
            return False
        
        # The above closure has a bug: used_mask is not updated correctly if we backtrack from a row.
        # We need to manage the state of `used_mask` carefully.
        
        # Revised approach for try_construct_with_limit:
        used_mask_list = [0]
        
        def build_row_safe(idx):
            if time.time() > deadline:
                return False
            
            if idx == n_rows:
                return True
            
            row = [0]
            
            def backtrack_pos(pos, current_max):
                if time.time() > deadline:
                    return False
                
                if pos == k + 1:
                    # Row complete
                    # Calculate mask for this row
                    row_mask = 0
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            row_mask |= (1 << d)
                    
                    # Check conflict with global
                    if (used_mask_list[0] & row_mask) != 0:
                        return False
                    
                    # Commit
                    old_mask = used_mask_list[0]
                    used_mask_list[0] |= row_mask
                    rows.append(row[:])
                    
                    if build_row_safe(idx + 1):
                        return True
                    
                    # Undo
                    used_mask_list[0] = old_mask
                    rows.pop()
                    return False
                
                start = current_max + 1
                end = M + 1
                
                candidates = []
                for x in range(start, end):
                    # Quick check: does x create a diff that conflicts with current row or global?
                    # We can check global here to prune early.
                    valid = True
                    for a in row:
                        d = x - a
                        if (used_mask_list[0] >> d) & 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check internal conflicts (duplicates in this row)
                    # Since row is increasing, diffs are unique if we check against all previous?
                    # No, a_j - a_i can equal a_k - a_l.
                    # We need to ensure all pairwise diffs in the new row are unique.
                    # We can maintain a set of diffs for the current row.
                    # But we are passing `row` which is a list.
                    # Let's pass a set of diffs for the current row as well?
                    # Or just recompute? For small k, recompute is fine.
                    
                    # Recompute all diffs for row + [x]
                    new_row = row + [x]
                    diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    
                    if valid:
                        # Heuristic: prefer larger diffs
                        score = sum(1.0/d for d in diffs)
                        candidates.append((score, x, diffs))
                
                if not candidates:
                    return False
                
                rng.shuffle(candidates)
                candidates.sort(key=lambda t: t[0])
                
                for score, x, diffs in candidates:
                    if backtrack_pos(pos + 1, x):
                        return True
                
                return False
            
            if backtrack_pos(1, 0):
                return True
            
            return False
        
        if build_row_safe(0):
            return rows
        return None

    # Main loop: Try increasing M
    # Start from lower_bound
    best_rows = None
    best_max = float('inf')
    
    # First, get a baseline with a simple greedy to have a valid solution
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    baseline_rows = pure_greedy()
    best_rows = baseline_rows
    best_max = max(max(r) for r in baseline_rows)
    
    # Now try to improve using the smarter backtracker
    # We try M values starting from lower_bound up to best_max - 1
    # If we find a solution with M, we update best_max and continue to see if we can go lower?
    # No, if we find a solution with M, we know M is achievable.
    # We want the smallest M.
    # So we should search M from lower_bound upwards.
    # If we find a solution at M, that is the optimal for that search?
    # Not necessarily, because the search might be incomplete (timeout).
    # But if we complete the search for M and fail, then M is not achievable.
    # If we succeed, M is achievable.
    # So we can iterate M from lower_bound to best_max - 1.
    
    current_seed = 1
    # We don't know the exact optimal M, so we start from lower_bound.
    # To save time, we can skip some M values if we are running out of time.
    
    for M in range(lower_bound, best_max):
        if time.time() > deadline - 10: # Reserve time for final check
            break
        
        # Try a few seeds for this M to increase chance of success
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            seed = current_seed
            current_seed += 1
            
            # Pass M to the constructor
            # We need to modify try_construct_with_limit to take M and seed
            # It currently uses `M` from the closure? No, I defined it inside.
            # Let's refactor to pass M.
            
            # Re-implementing the inner function with M as parameter is cleaner.
            
            def solve_for_M(M_val, seed_val):
                rng = random.Random(seed_val)
                used_mask_list = [0]
                rows = []
                
                def build_row_safe(idx):
                    if time.time() > deadline:
                        return False
                    
                    if idx == n_rows:
                        return True
                    
                    row = [0]
                    
                    def backtrack_pos(pos, current_max):
                        if time.time() > deadline:
                            return False
                        
                        if pos == k + 1:
                            row_mask = 0
                            for i in range(len(row)):
                                for j in range(i+1, len(row)):
                                    d = row[j] - row[i]
                                    row_mask |= (1 << d)
                            
                            if (used_mask_list[0] & row_mask) != 0:
                                return False
                            
                            old_mask = used_mask_list[0]
                            used_mask_list[0] |= row_mask
                            rows.append(row[:])
                            
                            if build_row_safe(idx + 1):
                                return True
                            
                            used_mask_list[0] = old_mask
                            rows.pop()
                            return False
                        
                        start = current_max + 1
                        end = M_val + 1
                        
                        candidates = []
                        for x in range(start, end):
                            valid = True
                            for a in row:
                                d = x - a
                                if (used_mask_list[0] >> d) & 1:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            new_row = row + [x]
                            diffs = set()
                            for i in range(len(new_row)):
                                for j in range(i+1, len(new_row)):
                                    d = new_row[j] - new_row[i]
                                    if d in diffs:
                                        valid = False
                                        break
                                    diffs.add(d)
                                if not valid:
                                    break
                            
                            if valid:
                                score = sum(1.0/d for d in diffs)
                                candidates.append((score, x, diffs))
                        
                        if not candidates:
                            return False
                        
                        rng.shuffle(candidates)
                        candidates.sort(key=lambda t: t[0])
                        
                        for score, x, diffs in candidates:
                            if backtrack_pos(pos + 1, x):
                                return True
                        
                        return False
                    
                    if backtrack_pos(1, 0):
                        return True
                    
                    return False
                
                if build_row_safe(0):
                    return rows
                return None

            result = solve_for_M(M, seed)
            if result is not None:
                best_rows = result
                best_max = M
                found = True
                break
        
        if found:
            # If we found a solution at M, we can stop if M == lower_bound
            if M == lower_bound:
                break
            # Otherwise, we continue to see if M-1 was possible? 
            # No, we are iterating upwards. If we found M, we know M is possible.
            # But we started at lower_bound. If lower_bound failed, we tried lower_bound+1, etc.
            # So the first M that succeeds is the minimum M for this specific search path.
            # However, the search is randomized. It might fail for a valid M due to bad luck.
            # So we should try multiple seeds for each M.
            # If we exhaust seeds for M and fail, we move to M+1.
            # If we succeed at M, we have a candidate. Can we do better?
            # Only if M > lower_bound. But we already tried lower M values.
            # So we can break.
            break

    return best_rows
# EVOLVE-BLOCK-END
