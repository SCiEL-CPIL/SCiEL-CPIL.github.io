import time
import math
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total distinct differences needed
    # Each row has k+1 elements, so (k+1)*k/2 differences per row.
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on max entry M:
    # Differences are in {1, ..., M}. We need total_diffs_needed distinct values.
    # So M >= total_diffs_needed.
    lower_bound_M = total_diffs_needed
    
    # Heuristic upper bound: usually close to lower bound or slightly higher.
    # The best known for (4,7) is 134. Lower bound is 4 * 28 = 112.
    # We will search M from lower_bound_M upwards.
    
    # To be safe and fast, we'll try to find a solution for a specific M.
    # If we find one, we return it.
    # We can try increasing M if the first few fail, but usually the first feasible M is close to optimal.
    # However, finding *any* solution for M=LB might be hard.
    # Let's start with M = lower_bound_M and go up.
    
    # For robustness, let's cap the search on M.
    max_M_to_try = lower_bound_M + 50
    
    # Precompute all possible differences for a given M
    # Actually, we just need to track used differences.
    
    # Strategy:
    # 1. Iterate M from lower_bound_M to max_M_to_try
    # 2. For each M, run a backtracking search to assign rows.
    # 3. Order rows: The "hardest" row to fill might be the one with the most constraints?
    #    Actually, all rows have the same structure. But the *available* differences change as we fill rows.
    #    A common heuristic for such packing problems is to fill rows in an order that maximizes constraints early.
    #    Since rows are identical in structure, we can just fill them one by one.
    #    However, the choice of values in row 0 affects row 1 heavily.
    #    
    #    Let's use a randomized restart backtracking with forward checking.
    #    We will try to construct rows one by one.
    #    For each row, we choose values a_1 ... a_k such that 0 < a_1 < ... < a_k <= M
    #    and all differences a_j - a_i are unused.
    
    # To speed up, we can precompute valid sets? No, M is small enough (<=160ish) that we can just search.
    
    # Let's define the backtracking function for a single row given a set of used differences.
    # It returns a list of values for the row, or None.
    
    def find_row(used_diffs, M, depth, current_row, path, timeout_check):
        """
        Recursively finds a valid row of length k+1 (starting with 0)
        such that all differences are not in used_diffs.
        Returns the list of values or None.
        """
        if time.time() > timeout_check:
            return None
            
        if len(current_row) == k + 1:
            return current_row[:]
            
        # Determine the next index to fill
        idx = len(current_row)
        
        # The next value must be greater than current_row[-1]
        # And less than or equal to M
        # Also, to leave room for the remaining k+1-len(current_row) elements,
        # we need to ensure there is enough "space" in terms of differences?
        # Simple bound: next_val <= M - (k + 1 - len(current_row))
        # Because we need strictly increasing integers.
        
        start_val = current_row[-1] + 1
        end_val = M - (k - idx) # Need k-idx more elements after this one
        
        if start_val > end_val:
            return None
            
        # Try values. Heuristic: Try values that produce "rare" differences?
        # Or just try all?
        # To break symmetry or find solutions faster, we can randomize the order slightly
        # or stick to a deterministic order (e.g., smallest first) which tends to pack tight.
        # Smallest first is usually good for minimizing M.
        
        # Let's try values from start_val to end_val
        # We can also try to skip values that create "popular" differences if we want,
        # but since we are checking against used_diffs, we just need unused ones.
        
        # Optimization: If we pick a value, we generate idx new differences.
        # We should check if any of these are in used_diffs.
        
        for val in range(start_val, end_val + 1):
            # Check differences with all previous elements
            valid = True
            new_diffs = []
            for prev_val in current_row:
                d = val - prev_val
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Recursive step
                current_row.append(val)
                # Add new diffs to used_diffs
                for d in new_diffs:
                    used_diffs.add(d)
                
                result = find_row(used_diffs, M, depth + 1, current_row, path, timeout_check)
                
                if result is not None:
                    return result
                else:
                    # Backtrack
                    for d in new_diffs:
                        used_diffs.remove(d)
                    current_row.pop()
                    
        return None

    # Main loop over M
    # We will try multiple random restarts for each M to escape local minima in the search space
    # (though backtracking is complete, the *order* of trying values affects speed).
    # Actually, standard backtracking is complete. If a solution exists for M, it will find it.
    # The issue is time. We need to prune effectively.
    
    # To improve performance, we can use a "most constrained variable" heuristic?
    # Here the variables are the row entries. The constraint is the global difference set.
    # Filling rows one by one is a natural decomposition.
    
    # Let's add a bit of randomness to the value selection order in the inner loop
    # to avoid getting stuck in slow branches, or just rely on the small M.
    # Given M is ~112-134, the branching factor is small.
    # Row 1: a1 in [1, M-7]. ~120 choices.
    # Row 2: a1 in [1, M-7]. But many will be pruned by Row 1's diffs.
    
    # Let's try M from lower_bound_M upwards.
    
    # To handle the case where backtracking is too slow for the first M,
    # we can limit the time per M and move to the next M if it fails.
    # But if M is too small, it will fail quickly (or prove infeasible).
    # If M is feasible, it should find a solution.
    
    # Let's use a time budget per M attempt.
    time_per_M = 10  # seconds
    
    for M in range(lower_bound_M, max_M_to_try + 1):
        if time.time() > deadline - 2:
            break
            
        # Try to find a solution for this M
        # We can try a few different random seeds/orders for the value selection
        # to increase the chance of finding a solution quickly if the search tree is skewed.
        
        # Actually, let's just do a deterministic search first.
        # If it's too slow, we might need to randomize.
        
        # For (4,7), M=112 is the bound. M=134 is known.
        # The gap is small.
        
        used_diffs = set()
        rows = []
        
        # We will fill n_rows rows
        success = True
        row_deadline = start_time + time_per_M
        
        for r in range(n_rows):
            row = [0]
            # Use a helper that respects the global deadline
            # The find_row function needs to check time.
            
            # We pass a mutable timeout object or just check inside
            # Let's modify find_row to take a deadline
            row_result = find_row(used_diffs, M, 0, row, [], row_deadline)
            
            if row_result is None:
                success = False
                break
            else:
                rows.append(row_result)
        
        if success:
            # Verify validity (just in case)
            all_diffs = set()
            valid = True
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                return rows
            else:
                # This should not happen if find_row is correct
                continue
        
        # If failed, try next M
        # We might also want to randomize the search order for the next attempt
        # But for now, let's just increment M.
        
    # If we didn't find a solution within the M range or time limit,
    # we should return a best-effort valid solution or a greedy one.
    # The problem statement says "Valid when...". If we return an invalid one, score is 0.
    # So we must ensure we return a VALID solution.
    # The greedy initialization in the original code always produces a valid solution,
    # just with a larger scope.
    
    # Let's fall back to a simple greedy construction to ensure validity.
    # This ensures we don't return an invalid result.
    
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    return rows
