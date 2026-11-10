import random
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences per row is (k+1)*k/2
    # Total differences for all rows is n_rows * (k+1)*k/2
    # The scope M must be at least the total number of differences (since they are distinct positive integers)
    # Actually, the differences are distinct positive integers from 1..M.
    # So M >= n_rows * (k+1)*k/2.
    
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on M
    # The largest difference in any row is at most M.
    # The sum of the largest differences? No.
    # Just use the counting bound: M >= total_diffs.
    # Also, the largest element in a row is the sum of some differences? No.
    # The largest element a_k is the max difference in that row.
    # So M is the max of all a_k.
    
    # Let's start with a safe upper bound and try to lower it.
    # A greedy construction usually yields a valid M.
    # Let's first do a quick greedy pass to get an initial valid M.
    
    def greedy_construct(target_m):
        """
        Attempts to construct n_rows rows with max value <= target_m.
        Returns (rows, success)
        """
        used_diffs = set()
        rows = []
        
        def gen_row(used, limit):
            row = [0]
            local_diffs = set()
            
            for _ in range(k):
                found = False
                # We want to pick the smallest valid v > row[-1] such that v <= limit
                # and {v - a for a in row} is disjoint from used and local_diffs
                # Also, we must ensure that we can complete the row within limit.
                # Simple heuristic: just pick the smallest valid v.
                
                # Optimization: Precompute forbidden values?
                # For v, the new diffs are v - row[0], ..., v - row[-1].
                # These must not be in used or local_diffs.
                
                # Let's scan v from row[-1] + 1 to limit
                for v in range(row[-1] + 1, limit + 1):
                    new_diffs = [v - a for a in row]
                    valid = True
                    for d in new_diffs:
                        if d in used or d in local_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for d in new_diffs:
                            local_diffs.add(d)
                        found = True
                        break
                
                if not found:
                    return None
            
            return row
        
        for i in range(n_rows):
            row = gen_row(used_diffs, target_m)
            if row is None:
                return None, False
            rows.append(row)
            for i_idx in range(len(row)):
                for j_idx in range(i_idx + 1, len(row)):
                    used_diffs.add(row[j_idx] - row[i_idx])
        
        return rows, True

    # Initial greedy attempt to find a valid M
    # Start with a reasonable upper bound. 
    # The theoretical lower bound is total_diffs. 
    # Let's try total_diffs first, then increase if it fails.
    # But greedy might fail even if a solution exists.
    
    # Let's do a more robust initial construction:
    # Use the backtracking solver directly with a generous bound to get a valid solution.
    # Then try to lower the bound.
    
    def backtrack_solve(target_m):
        """
        Uses backtracking to find a valid set of rows with max value <= target_m.
        Returns the list of rows or None.
        """
        used_diffs = set()
        rows = []
        
        # Heuristic: Order rows? Maybe construct the "hardest" row first?
        # Usually, the first row is the most constrained if we pick small numbers.
        # Let's just construct row by row.
        
        def gen_row_backtrack(used, limit, row_idx):
            """
            Recursively generate a row.
            row: list of current elements
            pos: current position in row (0..k)
            """
            if len(rows) == n_rows:
                return True
            
            # We are constructing row `row_idx`
            # If we are starting a new row, initialize it
            if not rows or len(rows) <= row_idx:
                # This logic is tricky if we interleave. 
                # Let's stick to sequential row construction.
                # Row 0, then Row 1, etc.
                pass
            
            # Actually, let's just write a specific backtracker for one row given the used set.
            # Then loop over rows.
            
            # This function should generate ONE row.
            # If it fails, the caller (loop) should handle backtracking?
            # No, backtracking across rows is complex. 
            # If row 0 is fixed, and row 1 fails, we might need to change row 0.
            # For simplicity and speed, let's do:
            # 1. Generate row 0.
            # 2. Generate row 1 given row 0.
            # 3. If fail, backtrack to row 0? 
            #    This requires storing states.
            
            # Given the time limit, a simpler strategy:
            # Randomized Restart with Greedy + Local Repair?
            # Or: Iterative Deepening A*?
            
            # Let's try a different angle:
            # The problem is small (7 rows, 7 elements).
            # Total elements: 49.
            # Total diffs: 147.
            # M is around 105-112.
            
            # Let's use a simple DFS with pruning for the whole set.
            # State: (current_row_idx, current_row, used_diffs)
            # This state space is huge.
            
            # Alternative: 
            # Construct rows one by one.
            # If a row fails, restart the whole process with a different random seed or different heuristic for the first row.
            # This is "Randomized Restart".
            
            # Let's implement a robust single-row generator with backtracking.
            # Then we just restart the whole thing if any row fails.
            
            pass
            
        # Let's implement the "Restart" strategy with a smarter row generator.
        
        def gen_row_with_backtrack(used, limit):
            """
            Generates a row of k+1 elements starting at 0, max <= limit.
            Uses backtracking to ensure completion.
            Returns row or None.
            """
            row = [0]
            local_diffs = set()
            
            def backtrack(pos):
                if pos == k + 1:
                    return True
                
                # Determine valid candidates for row[pos]
                # Must be > row[pos-1] and <= limit
                # Diffs: v - row[i] for i < pos must be in valid set
                
                # Heuristic: Try smallest first? Or most constrained?
                # Smallest first is usually good for packing.
                
                # Optimization: 
                # We can precompute which values are "bad" for each position?
                # Too complex. Just scan.
                
                # To speed up, we can check if there are enough remaining values?
                # For now, simple scan.
                
                # We need to ensure that the diffs v - row[i] are not in used or local_diffs.
                
                # Let's collect valid candidates
                candidates = []
                start_val = row[pos-1] + 1
                for v in range(start_val, limit + 1):
                    valid = True
                    for i in range(pos):
                        d = v - row[i]
                        if d in used or d in local_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                        # Optional: Limit candidates to avoid huge branching?
                        # If we pick the smallest, we might get stuck.
                        # If we pick randomly, we might explore more.
                        # Let's pick the first few and try them.
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    return False
                
                # Try candidates
                # Randomize order to escape local minima?
                # random.shuffle(candidates)
                
                for v in candidates:
                    # Tentatively add v
                    row.append(v)
                    new_local_diffs = []
                    for i in range(pos):
                        d = v - row[i]
                        new_local_diffs.append(d)
                        local_diffs.add(d)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    row.pop()
                    for d in new_local_diffs:
                        local_diffs.remove(d)
                
                return False
            
            return backtrack(1)
        
        # Main loop: Try to construct all rows
        # We will try multiple times with different randomizations if we add randomness to candidate selection.
        # For now, deterministic greedy-with-backtrack.
        
        # To add diversity, we can shuffle the order of rows? No, rows are identical in constraints.
        # We can shuffle the candidate selection in gen_row_with_backtrack.
        
        # Let's modify gen_row_with_backtrack to accept a random seed or use global random.
        
        def gen_row_random_backtrack(used, limit):
            row = [0]
            local_diffs = set()
            
            def backtrack(pos):
                if pos == k + 1:
                    return True
                
                candidates = []
                start_val = row[pos-1] + 1
                for v in range(start_val, limit + 1):
                    valid = True
                    for i in range(pos):
                        d = v - row[i]
                        if d in used or d in local_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                        if len(candidates) >= 10: # Limit branching
                            break
                
                if not candidates:
                    return False
                
                # Randomize order
                random.shuffle(candidates)
                
                for v in candidates:
                    row.append(v)
                    new_local_diffs = []
                    for i in range(pos):
                        d = v - row[i]
                        new_local_diffs.append(d)
                        local_diffs.add(d)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    row.pop()
                    for d in new_local_diffs:
                        local_diffs.remove(d)
                
                return False
            
            return backtrack(1)
        
        # Strategy:
        # 1. Try to find a solution with M = total_diffs (lower bound).
        # 2. If fail, increase M by 1 and try again.
        # 3. For each M, try a few random restarts.
        # 4. Return the best found.
        
        best_rows = None
        best_m = float('inf')
        
        # Start from a slightly higher bound to ensure we find *a* solution quickly, then lower it?
        # Or start from lower bound and go up?
        # Starting from lower bound is better for minimizing score.
        
        # Let's estimate a safe upper bound first to ensure we don't waste time if the lower bound is too tight.
        # Greedy usually works.
        greedy_rows, _ = greedy_construct(total_diffs + 20)
        if greedy_rows:
            best_rows = greedy_rows
            best_m = max(max(r) for r in best_rows)
        
        # Now try to improve by searching for lower M
        # We will search M from total_diffs up to best_m - 1
        # If we find a solution with M < best_m, update.
        
        # To be efficient, we can jump.
        # But the range is small (105 to 112).
        
        # Let's iterate M from total_diffs to best_m
        # For each M, try a few random restarts.
        
        # Time management:
        # We have ~115s.
        # Backtracking is fast for small k.
        
        current_m = total_diffs
        max_restarts_per_m = 20
        
        while current_m <= best_m and time.time() < deadline:
            found = False
            for restart in range(max_restarts_per_m):
                if time.time() > deadline:
                    break
                
                used_diffs = set()
                rows = []
                success = True
                
                for r_idx in range(n_rows):
                    row = gen_row_random_backtrack(used_diffs, current_m)
                    if row is None:
                        success = False
                        break
                    rows.append(row)
                    # Update used_diffs
                    for i_idx in range(len(row)):
                        for j_idx in range(i_idx + 1, len(row)):
                            used_diffs.add(row[j_idx] - row[i_idx])
                
                if success:
                    # Found a valid solution with max <= current_m
                    # The actual max might be less, but we are targeting current_m.
                    # We can compress it further? 
                    # Actually, if we found it with limit current_m, the max element is <= current_m.
                    # We can just return this? 
                    # Wait, we want the MINIMUM possible max.
                    # If we find a solution with limit current_m, it's valid.
                    # We can try to see if it works with a smaller limit? 
                    # No, the search was constrained to current_m.
                    # So this solution has max <= current_m.
                    # We found the smallest M that works (since we started from lower bound).
                    # So this is optimal (or close to it).
                    
                    # Let's verify the actual max
                    actual_max = max(max(r) for r in rows)
                    if actual_max < best_m:
                        best_m = actual_max
                        best_rows = rows
                        # Since we are iterating M upwards, and we found one at current_m,
                        # we can stop? 
                        # Yes, because any M < current_m failed (we tried them).
                        # So current_m is the minimum M for which a solution was found.
                        # The actual max of the solution might be < current_m, but that would imply
                        # a solution existed for a smaller M, which we would have found earlier.
                        # Unless the randomization failed to find it.
                        # So we should continue to lower M? No, we started from lower.
                        # So we can return immediately.
                        return rows
            
            current_m += 1
            
            # If we run out of time, return best found
            if time.time() > deadline:
                break
        
        if best_rows:
            return best_rows
        
        # Fallback: Return greedy result if any
        if greedy_rows:
            return greedy_rows
        
        # Last resort: Simple greedy without backtracking (should not happen)
        # Just return a valid set if we can find one quickly
        used = set()
        rows = []
        for i in range(n_rows):
            row = [0]
            for j in range(k):
                v = row[-1] + 1
                while True:
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used):
                        break
                    v += 1
                row.append(v)
                used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
