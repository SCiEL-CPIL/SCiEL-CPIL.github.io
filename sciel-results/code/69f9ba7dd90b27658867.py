import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Use 145s out of 150s to be safe against evaluator overhead
    deadline = start_time + 145.0
    
    # The task is (4, 7) difference triangle set.
    # n_rows = 4, k = 7.
    # Each row has k+1 = 8 elements.
    # Total differences per row: C(8, 2) = 28.
    # Total differences across rows: 4 * 28 = 112.
    # The differences must be distinct and positive integers.
    # The smallest possible set of differences is {1, 2, ..., 112}.
    # Therefore, the largest difference in any row must be at least 112.
    # Since the row starts at 0 and increases, the largest element in a row
    # is the difference between the last and first element (0).
    # So, the largest entry in the entire set (scope) must be at least 112.
    
    # The best known is 134.
    # We will try to find a solution with scope M starting from 112 upwards.
    # We use backtracking with forward checking.
    
    # Variables:
    # We can model this as assigning differences to rows, or assigning elements to rows.
    # Assigning elements is harder for forward checking on differences.
    # Assigning differences to (row, pair) slots is a classic exact cover / CSP approach.
    
    # Let's stick to element assignment but use strong pruning.
    # State:
    # rows: list of lists, partially filled.
    # used_diffs: set of all differences used so far.
    
    # Heuristic:
    # Order rows by difficulty? All rows are symmetric initially.
    # Order elements within a row: must be increasing.
    
    # A more powerful approach for this specific problem structure:
    # Since we need distinct differences, we can view this as packing 112 distinct differences
    # into 4 rows of 28 differences each.
    
    # Let's try a hybrid:
    # 1. Determine a target scope M.
    # 2. Use backtracking to construct rows one by one.
    # 3. When constructing a row, we need to choose 7 integers x_1 < x_2 < ... < x_7 such that
    #    all differences x_j - x_i are distinct and not in `used_diffs`.
    #    Also x_7 <= M.
    
    # To make this efficient, we can precompute valid "row patterns" for a given M?
    # No, M is small (112-134), but the number of subsets of size 7 from {1..M} is C(134, 7) ~ 10^10, too many.
    
    # Alternative: Backtrack on elements.
    # Row 0: Choose x_1, x_2, ..., x_7.
    #   x_1 must be >= 1.
    #   x_i must be > x_{i-1}.
    #   All diffs must be unique globally.
    
    # Optimization:
    # We can fix Row 0 to be the "most packed" or just search for any valid set.
    # Symmetry breaking: We can assume the rows are sorted by their largest element?
    # Or simply that the first element of the first row is the smallest possible?
    
    # Let's implement a recursive backtracking solver.
    
    def solve(M):
        # used_diffs: set of integers
        # rows: list of current rows
        # current_row_idx: which row we are filling
        # current_elem_idx: which element in the row we are filling
        
        # To speed up, we can use bitsets for used_diffs if M is small enough?
        # M ~ 134. 134 bits fits in 3 Python ints or a single int in C, but Python int is arbitrary precision.
        # Using a set of ints is fine for 112 items.
        
        used_diffs = set()
        rows = []
        
        # We will fill rows one by one.
        # For each row, we fill elements one by one.
        
        # To improve pruning, we can check if it's possible to complete the remaining rows.
        # Simple bound: The remaining differences needed are 28 * (n_rows - current_row_idx).
        # The available differences in range [1, M] not in used_diffs must be >= remaining_needed.
        # This is a weak bound since we don't know which diffs will be used.
        
        # Stronger bound:
        # For the current row being built, if we are at element j, we need to place k-j more elements.
        # The differences generated will be distinct.
        
        # Let's use a simple recursive search with MRV-like heuristics.
        # Actually, since the structure is fixed (rows of length 8), we just iterate.
        
        # To handle the time limit, we will interrupt if we exceed deadline.
        # We'll use a global flag or exception.
        
        class TimeoutError(Exception):
            pass
        
        def backtrack(row_idx, elem_idx, current_row):
            if time.time() > deadline:
                raise TimeoutError()
            
            if row_idx == n_rows:
                # All rows filled
                return True
            
            if elem_idx == k + 1:
                # Current row is complete
                rows.append(current_row[:])
                return backtrack(row_idx + 1, 0, [0])
            
            # We are choosing current_row[elem_idx]
            # Let prev = current_row[elem_idx - 1] if elem_idx > 0 else 0
            prev = current_row[elem_idx - 1] if elem_idx > 0 else 0
            
            # Max possible value for this element is M.
            # But it must be > prev.
            
            # Pruning:
            # The smallest possible difference we can generate for this element (if we pick x) is x - prev.
            # We need to ensure that we can still generate enough distinct differences for the rest of the row.
            
            # Let's just iterate x from prev + 1 to M.
            # To be smarter, we can skip x if x - prev is already in used_diffs.
            
            # Optimization:
            # If elem_idx is the last element (k), we just need to pick x such that
            # all diffs x - current_row[i] are not in used_diffs.
            
            for x in range(prev + 1, M + 1):
                # Check immediate differences with previous elements in this row
                # new_diffs = {x - a for a in current_row}
                # If any d in new_diffs is in used_diffs, skip.
                
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Also, we must ensure that the remaining elements can be placed.
                # This is hard to check exactly.
                
                # Apply move
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, elem_idx + 1, current_row):
                    return True
                
                # Undo move
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
            return False
        
        try:
            # Start with first row
            # Symmetry breaking: The first element of the first row can be assumed to be 1?
            # Not necessarily, but we can try small values first.
            # Actually, shifting a row by a constant doesn't change internal differences,
            # but it changes differences relative to 0 (which are included since row starts at 0).
            # Wait, the problem says "starting at 0". So the first element is fixed to 0.
            # So current_row always starts with 0.
            
            # Initial call: row 0, elem 1 (since elem 0 is 0), current_row = [0]
            if backtrack(0, 1, [0]):
                return rows
        except TimeoutError:
            return None
        
        return None

    # Search for the smallest M
    # Lower bound: 112
    # Upper bound: 134 (known best)
    
    # We will try M from 112 upwards.
    # To make it more robust, we can randomize the order of x values in the loop?
    # No, deterministic is better for finding the first solution.
    # However, backtracking might be very slow if it gets stuck in a bad branch.
    # We can add a "restart" mechanism: if no solution found for M in T seconds, try M+1?
    # Or just increase M.
    
    # Given the tight constraints, 112 is likely infeasible or very hard.
    # 113, 114...
    
    # Let's run the solver for M in range(112, 135).
    # If we find a solution, return it.
    
    # To avoid getting stuck on an infeasible M for the entire time limit,
    # we should limit the time per M.
    
    # Let's say we give 10 seconds per M value.
    # If M=112 takes 10s and fails, try M=113.
    
    # But the `backtrack` function checks `deadline` globally.
    # We need a per-attempt timeout.
    
    def solve_with_timeout(M, time_limit):
        start_t = time.time()
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, elem_idx, current_row):
            nonlocal start_t
            if time.time() > start_t + time_limit:
                raise TimeoutError()
            
            if row_idx == n_rows:
                return True
            
            if elem_idx == k + 1:
                rows.append(current_row[:])
                return backtrack(row_idx + 1, 0, [0])
            
            prev = current_row[elem_idx - 1] if elem_idx > 0 else 0
            
            for x in range(prev + 1, M + 1):
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, elem_idx + 1, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        try:
            if backtrack(0, 1, [0]):
                return rows
        except TimeoutError:
            pass
        
        return None

    # Try M from 112 to 134
    # Allocate time wisely.
    # Total time 145s.
    # 23 values of M. ~6s each.
    
    # However, if 112 is infeasible, it might exhaust its time quickly or slowly.
    # If it's infeasible, backtracking explores the whole tree, which can be huge.
    # So the timeout is crucial.
    
    # Let's start with a slightly higher M to ensure we get *a* solution, then try to lower it?
    # No, the score is the scope. We want the minimum.
    # So we must try 112 first.
    
    # But if 112 is infeasible, we waste time.
    # Is there a way to prove infeasibility faster?
    # Maybe use a SAT solver? Not available.
    
    # Let's just run it. If we find a solution at M, return it.
    # If we run out of time without finding any, we return the best known or a greedy solution.
    
    # We should keep a fallback greedy solution just in case.
    
    def greedy_solution():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # We'll run the search.
    # To maximize chances, we can interleave?
    # No, sequential M is logical.
    
    # Let's give more time to smaller M?
    # Or just uniform.
    
    # Let's try M=112 with 20s, M=113 with 20s, etc.
    # If we find one, stop.
    
    # Actually, if M=112 is infeasible, the search space is the same size regardless of M (mostly),
    # just the leaf condition changes.
    
    # Let's just loop.
    
    for M in range(112, 135):
        # Check if we still have time
        if time.time() > deadline - 5:
            break
        
        # Calculate remaining time
        remaining = deadline - time.time()
        # Give this M a portion of the remaining time, but cap it at 10s to avoid getting stuck
        # If there are many M left, give less.
        # If this is the last few M, give more.
        
        # Simple heuristic: 5 seconds per M, minimum 1s.
        t_limit = min(5.0, remaining / 2.0)
        
        sol = solve_with_timeout(M, t_limit)
        if sol:
            return sol
        
    # If no solution found with backtracking, return greedy
    # This should not happen if the time limit is sufficient for M=134.
    # Greedy scope is usually around 150-200.
    # But 134 is known to exist.
    # If backtracking fails for all M up to 134, it's a bug or timeout issue.
    
    return greedy_solution()
