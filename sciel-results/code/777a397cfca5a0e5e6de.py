import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Randomized Greedy Set Packing with Local Repair.
    1. Determine the target scope M (starting from lower bound).
    2. Pre-compute all valid "row signatures" (sets of differences) for scope M.
       A row signature is a set of diffs D that can be realized by a row 0=x0<...<xk.
       Actually, generating all is too expensive. Instead, we generate candidate rows 
       on the fly or use a heuristic generation.
       
    Refined Strategy: 
    We know the structure. A row is defined by its elements. 
    The set of differences is fixed by the elements.
    Instead of backtracking on elements, we use a "Greedy with Backtracking on Row Choice" approach.
    
    However, a more powerful structural change is to use **Simulated Annealing** or **Local Search** 
    on the *assignment of differences to rows*.
    
    Let's try a different structural family: **Integer Linear Programming (ILP) relaxation via Greedy**.
    Actually, ILP solvers are not in stdlib.
    
    Let's try **Randomized Conflict-Driven Backtracking**.
    1. Pick a scope M.
    2. Maintain a set of available differences.
    3. For each row, generate a random valid row using only available differences.
    4. If conflict (diff already used), backtrack or repair.
    
    The current program does element-by-element backtracking.
    The new strategy: **Row-Level Backtracking with Pre-computed Valid Rows**.
    Pre-computing all valid rows for M=112 is feasible?
    Number of rows is huge.
    
    Let's look at Inspiration 1: CSP + Randomized Local Search.
    Let's look at Inspiration 2: Simulated Annealing.
    
    I will implement a **Greedy Construction with Randomized Restart and Conflict Resolution**.
    1. Start with M = lower_bound.
    2. Attempt to construct rows one by one.
    3. For each row, use a heuristic to pick a valid row from the remaining differences.
       Heuristic: Pick a row that "uses up" differences that are "scarce" or "hard to place".
    4. If stuck, restart with a different random seed or slightly larger M.
    
    To make this robust, I will implement a **Randomized Greedy** that generates candidate rows 
    by sampling valid structures.
    
    Key Insight: A valid row of length k+1 is determined by its k elements x1..xk.
    The differences are {xj-xi}.
    We can generate a random valid row within scope M by picking random x1..xk 
    and checking if all differences are within M and distinct.
    But we need them to be distinct *globally*.
    
    New Structural Approach: **Difference-Pool Consumption via Randomized Greedy**.
    1. Initialize pool of diffs {1..M}.
    2. For each row i:
       a. Generate a candidate row R using only diffs in pool.
          How? 
          - Pick x1..xk randomly? 
          - Better: Pick the largest diff in the pool as xk? No, xk is just the max element.
          - The set of diffs D_R must be a subset of pool.
          - We can generate a random row, compute its diff set D_R.
          - If D_R is a subset of pool, accept it, remove D_R from pool.
          - If not, reject and try another.
       b. If we fail to find a row after N tries, fail.
    3. If all rows found, return solution.
    4. If fail, increase M or restart.
    
    This is structurally different from element-by-element backtracking because it treats the row as an atomic unit and uses random sampling of the row space, which can be more efficient if the space of valid rows is dense enough for the given M.
    """
    start_time = time.time()
    DEADLINE = 145.0

    num_elements = k + 1
    diffs_per_row = num_elements * (num_elements - 1) // 2

    # Lower bound calculation
    # Total diffs = n_rows * diffs_per_row
    # Max diff <= M
    # Number of available diffs in [1, M] is M.
    # So M >= n_rows * diffs_per_row
    total_diffs_needed = n_rows * diffs_per_row
    lower_bound = total_diffs_needed
    
    # Known best for (7,6) is 112. 
    # We will try M from lower_bound to lower_bound + 10.
    
    def is_timeup():
        return time.time() > DEADLINE

    def generate_random_row(M, used_diffs):
        """
        Generate a random row of length num_elements starting at 0, max <= M,
        such that all its differences are NOT in used_diffs.
        Returns the row list or None.
        """
        # We need to pick x1, ..., x_{k} such that 0 < x1 < ... < xk <= M
        # and all diffs xj-xi are not in used_diffs.
        
        # Heuristic: Try to pick values that are "sparse" to avoid collisions?
        # Or just random sampling.
        
        # Optimization: 
        # The differences involving xk are xk, xk-x1, ..., xk-x_{k-1}.
        # These are k diffs.
        # If we pick xk first, we constrain the others.
        
        # Let's try a randomized backtracking on the elements, but limited depth 
        # and with a "give up" condition to act as a sampler.
        
        def backtrack(pos, current_row, current_max):
            if is_timeup():
                return None
            
            if pos == num_elements:
                # Check global validity
                new_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in used_diffs:
                            return None
                        new_diffs.add(d)
                return current_row
            
            min_val = current_row[-1] + 1 if current_row else 1
            max_val = M - (num_elements - pos - 1)
            
            if min_val > max_val:
                return None
            
            # Randomly shuffle candidates to explore different parts of the space
            candidates = list(range(min_val, max_val + 1))
            random.shuffle(candidates)
            
            # Limit the number of candidates to try in this "sampling" phase
            # to avoid getting stuck in deep backtracking for a single row.
            # If we limit this, it becomes a "randomized greedy" for the row.
            limit = 20 
            
            for val in candidates[:limit]:
                # Quick check: does val create any diff with existing elements that is in used_diffs?
                valid = True
                for x in current_row:
                    d = val - x
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Recurse
                res = backtrack(pos + 1, current_row + [val], val)
                if res is not None:
                    return res
            
            return None
        
        return backtrack(1, [0], 0)

    def solve_with_scope(M):
        """
        Try to construct n_rows rows with max entry <= M.
        Uses randomized greedy row-by-row construction.
        """
        used_diffs = set()
        rows = [None] * n_rows
        
        # Order rows by some heuristic? 
        # Since rows are symmetric, we can just do 0..n_rows-1.
        # But maybe we should pick the "hardest" row first? 
        # In this symmetric problem, all rows are identical in constraint.
        
        for i in range(n_rows):
            if is_timeup():
                return None
            
            # Try to generate a valid row
            # Use multiple attempts for this specific row
            row = None
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts:
                if is_timeup():
                    return None
                row = generate_random_row(M, used_diffs)
                if row is not None:
                    break
                attempts += 1
            
            if row is None:
                return None
            
            # Update used_diffs
            for a_idx in range(len(row)):
                for b_idx in range(a_idx + 1, len(row)):
                    d = row[b_idx] - row[a_idx]
                    used_diffs.add(d)
            
            rows[i] = row
        
        return rows

    # Main loop
    # Try M from lower_bound to lower_bound + 10
    # With multiple restarts for each M
    
    best_solution = None
    best_M = float('inf')
    
    # We know 112 works for (7,6). 
    # Let's iterate M.
    for M in range(lower_bound, lower_bound + 10):
        if is_timeup():
            break
        
        found = False
        # Number of restarts depends on M. 
        # For small M, it's harder, so maybe more restarts? 
        # But time is limited. Let's do a fixed number of restarts per M, 
        # or a time-budgeted loop.
        
        restarts = 0
        while not found and not is_timeup():
            restarts += 1
            # Limit restarts per M to avoid wasting time on impossible M
            if restarts > 10:
                break
            
            res = solve_with_scope(M)
            if res is not None:
                # Verify
                all_diffs = []
                valid = True
                for r in res:
                    if r is None:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            all_diffs.append(r[j] - r[i])
                
                if len(all_diffs) == len(set(all_diffs)) and len(all_diffs) == n_rows * diffs_per_row:
                    best_solution = res
                    best_M = M
                    found = True
                else:
                    # Should not happen if generate_random_row is correct, but just in case
                    pass
        
        if found:
            return best_solution

    # Fallback: If no solution found with the greedy method, use a simpler greedy 
    # that might have a larger scope but is guaranteed to terminate quickly.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows
