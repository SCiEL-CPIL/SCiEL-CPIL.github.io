import sys
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Constants for DTS(4, 7)
    n = n_rows
    k_val = k
    
    # We know the answer is around 134.
    # Lower bound is 112.
    
    def try_find_solution(M, time_limit):
        """
        Tries to find a DTS with scope M.
        Returns list of rows or None.
        """
        used_mask = 0
        rows = []
        
        # To make the search faster, we can order the rows.
        # It's often easier to build the "most constrained" rows first?
        # Or just build them sequentially.
        
        def solve_row(idx, current_mask, current_rows):
            if idx == n:
                return True
            
            # Find a valid row
            row_found = find_row(idx, current_mask, M)
            if row_found is None:
                return False
            
            # Update mask and rows
            new_mask = current_mask
            for d in row_found['diffs']:
                new_mask |= (1 << d)
                
            current_rows.append(row_found['vals'])
            
            if solve_row(idx + 1, new_mask, current_rows):
                return True
            
            # Backtrack
            current_rows.pop()
            return False
        
        def find_row(idx, mask, M):
            # We need to pick k_val values v_1..v_k from 1..M
            # such that all diffs are unique and not in mask.
            
            # We use backtracking to build the row.
            # vals = [0]
            # We pick v_1, then v_2, etc.
            
            # To improve performance, we can use a heuristic:
            # Try values in a randomized order?
            # Or just increasing order?
            # Increasing order is deterministic. Randomized might escape local minima faster.
            
            # Let's use a recursive helper.
            # To avoid deep recursion overhead, we can use an iterative stack or just recursion.
            # Depth is 7, so recursion is fine.
            
            def backtrack(pos, last_val, current_vals, local_mask):
                if pos == k_val:
                    return True
                
                # Try candidates for current_vals[pos]
                # Candidates must be > last_val and <= M
                # To diversify, we can shuffle candidates?
                # But we need to be careful with time.
                
                # Heuristic: Try values that create "new" differences efficiently.
                # Let's just try increasing order first, as it's simple.
                # If it's too slow, we add randomization.
                
                for v in range(last_val + 1, M + 1):
                    # Check if adding v creates any conflict
                    # Diffs with all previous values in current_vals (including 0)
                    conflict = False
                    new_diffs = []
                    for u in current_vals:
                        d = v - u
                        if d > M: # Should not happen if v <= M and u >= 0
                            conflict = True
                            break
                        # Check if d is already in local_mask or global mask
                        if (local_mask >> d) & 1 or (mask >> d) & 1:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    # No conflict, recurse
                    new_local_mask = local_mask
                    for d in new_diffs:
                        new_local_mask |= (1 << d)
                    
                    if backtrack(pos + 1, v, current_vals + [v], new_local_mask):
                        return True
            
            # Start backtracking
            # current_vals starts with [0]
            if backtrack(0, 0, [0], 0):
                # We need to return the actual values and diffs
                # The backtrack function above returns True/False but doesn't return the values.
                # We need to modify it to return the row.
                pass
            
            # Let's rewrite find_row to properly return the row.
            return None
        
        # The above pseudo-code is a bit messy. Let's implement a cleaner version.
        
        def find_row_clean(idx, mask, M, rng):
            # Returns (vals, diffs_mask) or None
            
            # We will use a list to hold the current row construction
            # We try to build the row.
            
            # To speed up, we can pre-filter candidates for the first element?
            # v_1 must be such that v_1 is not in mask.
            
            def rec(pos, last, vals, l_mask):
                if pos == k_val:
                    return vals, l_mask
                
                # Get candidates
                # To improve search, we can randomize the start or step?
                # Let's try increasing order.
                
                for v in range(last + 1, M + 1):
                    # Check conflicts
                    ok = True
                    new_diffs = []
                    for u in vals:
                        d = v - u
                        if (l_mask >> d) & 1 or (mask >> d) & 1:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    new_l_mask = l_mask
                    for d in new_diffs:
                        new_l_mask |= (1 << d)
                    
                    res = rec(pos + 1, v, vals + [v], new_l_mask)
                    if res:
                        return res
                
                return None
            
            res = rec(0, 0, [0], 0)
            if res:
                vals, l_mask = res
                # vals[0] is 0
                # Return vals[1:] as the row entries (excluding 0)
                # The problem asks for "list of k+1 integers that starts at 0"
                # So we return vals
                return vals, l_mask
            return None

        # Main solve loop
        rng = random.Random(42)
        
        def solve(idx, mask, rows):
            if idx == n:
                return True
            
            # Try to find a row
            # We can try multiple times with different random seeds if we add randomization
            # For now, deterministic increasing order.
            
            res = find_row_clean(idx, mask, M, rng)
            if res is None:
                return False
            
            vals, l_mask = res
            new_mask = mask | l_mask
            rows.append(vals)
            
            if solve(idx + 1, new_mask, rows):
                return True
            
            rows.pop()
            return False
        
        rows = []
        if solve(0, 0, rows):
            return rows
        return None

    # Iterate M from lower bound to upper bound
    # Lower bound: 112
    # Upper bound: 134 (known best) + some margin
    
    # To be safe, let's start from 112 and go up.
    # But if the search for 112 is very slow, we waste time.
    # We can set a time limit per M.
    
    best_sol = None
    
    # Estimate time per M. If we have 140s, and we try 20 values of M, we have 7s per M.
    # That's plenty for backtracking with bitsets for small k.
    
    for M in range(112, 140):
        if time.time() > deadline:
            break
            
        # Add a small time limit per M to avoid getting stuck on a hard unsolvable M
        # e.g., 2 seconds per M
        t_start = time.time()
        
        # We can add randomization to find_row to escape local minima.
        # Let's modify find_row_clean to use randomization in candidate ordering.
        
        def find_row_random(idx, mask, M, rng):
            def rec(pos, last, vals, l_mask):
                if pos == k_val:
                    return vals, l_mask
                
                # Generate candidates
                candidates = list(range(last + 1, M + 1))
                # Shuffle?
                # Shuffling 100+ items is cheap.
                # But maybe we want to bias towards smaller values?
                # Or just random.
                rng.shuffle(candidates)
                
                for v in candidates:
                    ok = True
                    new_diffs = []
                    for u in vals:
                        d = v - u
                        if (l_mask >> d) & 1 or (mask >> d) & 1:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    new_l_mask = l_mask
                    for d in new_diffs:
                        new_l_mask |= (1 << d)
                    
                    res = rec(pos + 1, v, vals + [v], new_l_mask)
                    if res:
                        return res
                
                return None
            
            res = rec(0, 0, [0], 0)
            if res:
                vals, l_mask = res
                return vals, l_mask
            return None

        def solve_random(idx, mask, rows, rng):
            if idx == n:
                return True
            
            res = find_row_random(idx, mask, M, rng)
            if res is None:
                return False
            
            vals, l_mask = res
            new_mask = mask | l_mask
            rows.append(vals)
            
            if solve_random(idx + 1, new_mask, rows, rng):
                return True
            
            rows.pop()
            return False

        # Try a few random restarts for this M
        found = False
        for attempt in range(5):
            if time.time() - t_start > 2.0: # 2s limit per attempt
                break
            if time.time() > deadline:
                return best_sol
            
            rng = random.Random(attempt)
            rows = []
            if solve_random(0, 0, rows, rng):
                best_sol = rows
                found = True
                break
        
        if found:
            # We found a solution for this M.
            # Since we are iterating M from low to high, this is the best we can do for this M.
            # Can we do better with a smaller M? No, we started from 112.
            # So we can return immediately?
            # Yes, because we want the *minimum* scope.
            # If we found a solution for M, any larger M is worse.
            # So we return the first M that works.
            return best_sol

    # If no solution found in the loop (should not happen for M up to 134)
    # Fall back to a greedy solution to ensure we return *something*
    # Greedy:
    used = set()
    rows = []
    for _ in range(n):
        row = [0]
        x = 0
        while len(row) < k_val + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
