import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    def check_solution(rows):
        """Verify validity and return scope."""
        used = set()
        max_val = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return None, -1 # Invalid
                    used.add(d)
            if row[-1] > max_val:
                max_val = row[-1]
        return rows, max_val

    def solve_csp(M, deadline, rng):
        """
        Try to find a solution where max element <= M.
        Returns list of rows or None.
        Uses randomized backtracking.
        """
        used_diffs = set()
        rows = []
        
        # State for backtracking
        # We construct rows one by one.
        # Within a row, we construct elements one by one.
        
        # To improve performance, we can use a recursive helper that manages the state.
        # But Python recursion limits and overhead might be an issue if we restart often.
        # However, depth is only n_rows * (k+1) which is small (4*8=32).
        
        def backtrack_row(row_idx, current_row):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete, move to next
                return backtrack_row(row_idx + 1, [0])
            
            last_val = current_row[-1]
            
            # Determine search range for next value
            # Must be > last_val and <= M
            # We want to explore values that don't conflict.
            
            # Optimization: 
            # Instead of trying every integer, we can sample.
            # But for correctness of "exists", we need to be thorough or use heuristics.
            # Given the time limit, we use a "Limited Discrepancy" or simple random restart strategy.
            
            # Strategy: Pick a random candidate from a window, or scan sequentially with random skip.
            # Let's scan sequentially from last_val + 1 to M, but with a random probability to "skip" ahead?
            # No, that's not standard.
            
            # Better: Collect valid candidates in a window, pick one randomly.
            # Window size increases if we get stuck.
            
            # To make it efficient, we'll just iterate x from last_val+1 to M.
            # For each x, check if {x - a for a in current_row} is disjoint from used_diffs AND internal diffs are unique.
            # Wait, internal diffs of current_row are already unique by construction.
            # We just need to check new diffs against used_diffs.
            
            # To avoid iterating up to M (which might be 134) every time, 
            # we can observe that if M is small, the search is tight.
            
            found = False
            # Randomize the order of checking to avoid deterministic traps
            # We'll generate a list of candidates? 
            # If M is 134 and last is 0, that's 134 candidates.
            # Checking 134 candidates * 7 diffs = ~1000 ops. Very fast.
            # So we can just shuffle the candidates or pick randomly.
            
            # Let's pick candidates in a randomized order, biased towards smaller values to keep scope low?
            # Actually, for the decision problem (max <= M), we just need *any* valid completion.
            # Random order is good for diversity.
            
            candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                # Check against global used_diffs
                if not (new_diffs & used_diffs):
                    candidates.append(x)
            
            if not candidates:
                return False
            
            # Shuffle candidates to randomize search path
            rng.shuffle(candidates)
            
            for x in candidates:
                # Apply
                new_diffs = {x - a for a in current_row}
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                if backtrack_row(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
                if time.time() > deadline:
                    return False
            
            return False

        # Start backtracking
        if backtrack_row(0, [0]):
            return rows # Wait, rows is not updated in backtrack_row correctly in this sketch.
                       # I need to pass rows or return the constructed rows.
        
        return None

    # The above sketch has a flaw: `rows` is not filled correctly because `current_row` is local.
    # I need to restructure to store the rows found.
    
    def solve_csp_v2(M, deadline, rng):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:]) # Store a copy
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Find valid candidates
            # To speed up, we can limit the search window if M is large?
            # No, M is the bound.
            
            candidates = []
            # Optimization: If we are close to the end of the row, 
            # the remaining elements must fit in [last_val+1, M].
            
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
            
            if not candidates:
                return False
            
            # Randomize
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False

        if backtrack(0, [0]):
            return rows
        return None

    # Binary Search for the minimal M
    # Lower bound: counting bound is 112.
    # Upper bound: known best is 134. Let's start a bit higher, say 150.
    low = 112
    high = 150
    
    best_solution = None
    best_scope = float('inf')
    
    # We can just try values from low to high, or binary search.
    # Binary search is better.
    
    # However, CSPs are not monotonic in a way that makes binary search trivial?
    # Actually, if a solution exists for M, it exists for M+1 (same solution).
    # So it is monotonic.
    
    # But randomized solvers might fail to find a solution for M even if it exists, due to time limit/randomness.
    # So we should try multiple times for each M.
    
    # Strategy:
    # 1. Try to find a solution with M = low.
    # 2. If found, record it.
    # 3. If not found, try M+1, etc.
    # Or Binary Search:
    # Check mid. If found, high = mid. Else low = mid + 1.
    # But "found" is probabilistic.
    
    # Let's do a linear scan from low to high, but with a time budget per M.
    # Or, simpler: Just run the solver with M = 134 (known feasible) first to get a baseline.
    # Then try to reduce M.
    
    # Let's try a few random restarts for M = 134 first.
    # Then try 133, 132...
    
    # To be safe and fast:
    # 1. Run solver for M=134. If success, save.
    # 2. If time remains, try M=133.
    # 3. ...
    
    # Actually, the counting bound is 112. It's possible the answer is lower.
    # Let's scan from 112 upwards.
    
    rng = random.Random(42) # Fixed seed for reproducibility in initial attempts? 
    # No, random restarts are better.
    
    # We will iterate M from 112 to 140.
    # For each M, we give it a small time slice (e.g., 5s).
    # If we find a solution, we update best and continue to lower M? No, we are going up.
    # We want the *lowest* M.
    # So we check 112. If no solution, check 113.
    # This is inefficient if the answer is 134.
    
    # Better: Binary Search.
    # But we need to be sure.
    # Let's do this:
    # 1. Find ANY solution (M=150).
    # 2. Then binary search between 112 and found_M.
    
    # Step 1: Find a feasible solution
    current_best_rows = None
    current_best_M = 150
    
    # Try M=150
    sol = solve_csp_v2(150, start_time + 10, rng)
    if sol:
        current_best_rows = sol
        current_best_M = 150
    else:
        # If we can't even find 150, something is wrong or time is tight.
        # Try a few more times.
        for _ in range(5):
            if time.time() > deadline - 5: break
            sol = solve_csp_v2(150, time.time() + 2, rng)
            if sol:
                current_best_rows = sol
                current_best_M = 150
                break
    
    if not current_best_rows:
        # Fallback to greedy if CSP fails completely
        # (Unlikely, but safe)
        # ... implement greedy fallback ...
        pass

    # Step 2: Binary Search for minimal M
    if current_best_rows:
        low = 112
        high = current_best_M
        
        while low < high and time.time() < deadline - 5:
            mid = (low + high) // 2
            found = False
            # Try to find solution for M=mid
            # Give it a few seconds
            time_slice_end = time.time() + 3
            for attempt in range(3):
                if time.time() > time_slice_end: break
                sol = solve_csp_v2(mid, time.time() + 1, rng)
                if sol:
                    found = True
                    current_best_rows = sol
                    current_best_M = mid
                    break
            
            if found:
                high = mid
            else:
                low = mid + 1
        
        # Final check on low
        # We might have found a solution for 'high' which is now 'low' if it worked.
        # The loop ends when low == high.
        # If 'high' was updated, we have a solution for 'high'.
        # If not, we have a solution for the previous 'high'.
        # We should return the best solution found.
        
        # To ensure we return the valid one, we rely on current_best_rows being updated only on success.
        
    if current_best_rows:
        return current_best_rows
    else:
        # Fallback: Return a valid greedy solution if possible
        # Re-implementing the greedy from before as a last resort
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows
