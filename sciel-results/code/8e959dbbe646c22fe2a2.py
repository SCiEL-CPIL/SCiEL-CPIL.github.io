"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Parameters for the CSP solver
    # We are looking for n_rows rows of k+1 integers starting at 0.
    # Let the row be [0, x1, x2, ..., xk].
    # We need to assign x1..xk such that all differences are distinct globally.
    
    # Global state
    used_diffs = set()
    rows = []
    
    # Helper to check if a candidate value x is valid for a row given current partial row
    def is_valid_value(row, x, used_diffs):
        new_diffs = set()
        for a in row:
            d = x - a
            if d <= 0:
                return False
            if d in used_diffs:
                return False
            if d in new_diffs:
                return False
            new_diffs.add(d)
        return True, new_diffs

    # CSP Solver for a single row
    # row: current list of integers (starts with [0])
    # target_len: k+1
    # used_diffs: global set of used differences
    # max_val_limit: optional limit on the maximum value to enforce a bound
    def solve_row_csp(row, target_len, used_diffs, max_val_limit=None):
        if len(row) == target_len:
            return True
        
        last_val = row[-1]
        # Lower bound for next value
        next_start = last_val + 1
        
        # If we have a max_val_limit, we must ensure we can fit the remaining elements
        # Simple bound: if max_val_limit is set, x must be <= max_val_limit
        # Also, we need enough room to place (target_len - len(row) - 1) more numbers
        # So x <= max_val_limit - (target_len - len(row) - 1)
        
        # Heuristic: Try values that are "tight" first or use MRV?
        # For difference sets, smaller values are often more constrained.
        # Let's iterate from smallest to largest to find a valid solution quickly.
        # To allow for search, we can randomize the order slightly or use backtracking.
        
        # Optimization: Pre-compute possible candidates?
        # Since k is small (e.g., 5 for k=5, so 6 elements), the row length is small.
        
        # Range of x
        # Upper bound: If max_val_limit is None, we can go up to some heuristic limit.
        # A safe upper bound for a single row in a difference set is roughly O(k^2) or similar,
        # but since we are optimizing for global max, we should try to keep values small.
        # However, to be safe, we can set a dynamic upper bound.
        # The counting bound is 105 for (7,5). Let's use a limit slightly above the current best known or a heuristic.
        # For the solver, if no limit is provided, we cap it to avoid infinite loops, 
        # but we want to find the *smallest* possible row.
        
        # Let's try values in increasing order.
        # If we fail, backtracking will happen at a higher level.
        
        # To make it a true CSP with backtracking, we need to explore alternatives.
        # We'll use a recursive search.
        
        # Heuristic: Sort candidates by how many conflicts they create? 
        # For now, simple increasing order is a good baseline for "smallest max".
        
        # We need to bound the search space.
        # If max_val_limit is None, we assume we are in a phase where we are building a row 
        # without a strict global cap, but we still want small values.
        # We can set a local cap: last_val + k*10 or something? 
        # Actually, if we are trying to minimize the max, we should try small values first.
        
        # Let's define an upper bound for x in this call.
        # If we are in the middle of solving, we don't know the final max.
        # But we can set a heuristic upper bound: 200 is usually enough for small k.
        # If max_val_limit is provided, use it.
        if max_val_limit is None:
            # Heuristic upper bound for a single row element
            # We can refine this, but 200 is safe for k<=5, n_rows<=7
            upper_bound = 200 
        else:
            upper_bound = max_val_limit
            
        # We need to ensure we can complete the row.
        # Remaining elements to place: rem = target_len - len(row) - 1
        # So x must be <= upper_bound - rem
        
        effective_upper = upper_bound - (target_len - len(row) - 1)
        
        for x in range(next_start, effective_upper + 1):
            valid, new_diffs = is_valid_value(row, x, used_diffs)
            if valid:
                # Recurse
                new_row = row + [x]
                new_used = used_diffs | new_diffs
                if solve_row_csp(new_row, target_len, new_used, max_val_limit):
                    # Success, update global state via reference or return
                    row[:] = new_row
                    used_diffs.update(new_diffs)
                    return True
        
        return False

    # Main strategy:
    # 1. Try to build rows one by one using CSP.
    # 2. If a row fails to be built with a certain max limit, increase the limit or backtrack row order.
    # 3. To minimize the global max, we can use an iterative deepening approach on the max value M.
    #    Start with M = lower_bound, try to construct all rows with max entry <= M.
    #    If successful, return. If not, increment M.
    
    # Lower bound for M:
    # Counting bound: total differences = n_rows * k*(k+1)/2
    # Each difference d in 1..M can be used at most once.
    # So M * (M+1) / 2 >= n_rows * k*(k+1)/2
    # M^2 + M - n_rows*k*(k+1) >= 0
    # M >= (-1 + sqrt(1 + 4*n_rows*k*(k+1))) / 2
    
    import math
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound_M = int(math.ceil((-1 + math.sqrt(1 + 4 * total_diffs)) / 2))
    
    # We know the best known is 112 for (7,5).
    # Let's start searching from lower_bound_M upwards.
    
    best_rows = None
    best_max = float('inf')
    
    # To handle backtracking across rows, we can use a simple approach:
    # Try to build row 0, then row 1, etc.
    # If row i fails, we might need to restart row i-1 with different choices.
    # This is complex. A simpler robust approach for small N:
    # Use the CSP solver for each row sequentially, but if a row fails, 
    # we can try to "repair" or restart the entire construction with a slightly higher M.
    # However, to truly minimize M, we should try to find *any* valid configuration for a given M.
    
    # Let's implement a function that tries to build all rows with max <= M.
    def try_build_all_rows(M):
        nonlocal used_diffs, rows
        used_diffs = set()
        current_rows = []
        
        # We will build rows sequentially.
        # To allow for some flexibility, we can randomize the order of rows? 
        # No, rows are identical in constraints except for the used diffs.
        # However, the *order* in which we pick rows might matter for the CSP search.
        # Let's try building them in order 0..n_rows-1.
        
        for r in range(n_rows):
            row = [0]
            # Use CSP solver with max_val_limit = M
            if not solve_row_csp(row, k + 1, used_diffs, M):
                return False
            current_rows.append(row)
        
        return True, current_rows

    # Iterative deepening on M
    # Start from lower_bound_M
    # We can also try a few random restarts for each M if the first attempt fails,
    # by randomizing the order of candidate values in the CSP solver.
    
    # To add randomness to CSP:
    # We can modify solve_row_csp to shuffle candidates or pick randomly from valid ones.
    # But for minimizing M, we want the *first* valid solution found for a given M.
    # If we find one, M is achievable.
    
    # Let's add a "randomness factor" to the CSP solver to escape local dead-ends
    # within a fixed M.
    
    def solve_row_csp_randomized(row, target_len, used_diffs, max_val_limit, randomize=False):
        if len(row) == target_len:
            return True
        
        last_val = row[-1]
        next_start = last_val + 1
        
        if max_val_limit is None:
            upper_bound = 200
        else:
            upper_bound = max_val_limit
            
        rem = target_len - len(row) - 1
        effective_upper = upper_bound - rem
        
        candidates = list(range(next_start, effective_upper + 1))
        
        if randomize:
            # Shuffle to explore different branches
            random.shuffle(candidates)
        else:
            # Try smallest first to minimize local max
            pass
            
        for x in candidates:
            valid, new_diffs = is_valid_value(row, x, used_diffs)
            if valid:
                new_row = row + [x]
                new_used = used_diffs | new_diffs
                if solve_row_csp_randomized(new_row, target_len, new_used, max_val_limit, randomize):
                    row[:] = new_row
                    used_diffs.update(new_diffs)
                    return True
        
        return False

    def try_build_all_rows_randomized(M, num_restarts=5):
        nonlocal used_diffs, rows
        for restart in range(num_restarts):
            used_diffs = set()
            current_rows = []
            success = True
            for r in range(n_rows):
                row = [0]
                # Use randomized CSP for exploration
                if not solve_row_csp_randomized(row, k + 1, used_diffs, M, randomize=(restart > 0)):
                    success = False
                    break
                current_rows.append(row)
            
            if success:
                return True, current_rows
        return False, None

    # Search for the smallest M
    # We'll check M from lower_bound_M up to a reasonable limit (e.g., 150)
    # We'll use a time check to avoid running too long.
    
    current_M = lower_bound_M
    max_M_to_check = 150 # Safe upper bound
    
    while current_M <= max_M_to_check:
        if time.time() > deadline:
            break
            
        # Try to build with this M
        # First try deterministic (smallest values first)
        success, res_rows = try_build_all_rows_randomized(current_M, num_restarts=1)
        
        if success:
            best_rows = res_rows
            best_max = current_M
            break
        
        # If deterministic fails, try a few randomized restarts for this M
        # This helps if the deterministic order leads to a dead end but another order works.
        success, res_rows = try_build_all_rows_randomized(current_M, num_restarts=10)
        
        if success:
            best_rows = res_rows
            best_max = current_M
            break
        
        current_M += 1
        
        # If we've gone past the known best (112 for 7,5), we can relax the search
        # or just keep going until deadline.
        
    if best_rows is None:
        # Fallback: If we didn't find a solution within the M range, 
        # use the greedy method as a last resort to ensure we return *something* valid.
        # This is unlikely to be better than the CSP result, but better than crashing.
        used = set()
        greedy_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            greedy_rows.append(row)
        best_rows = greedy_rows
        best_max = max(max(r) for r in best_rows)
        
    return best_rows
# EVOLVE-BLOCK-END
