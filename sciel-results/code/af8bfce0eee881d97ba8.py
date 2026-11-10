"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Total number of unique differences required
    # Each row has k+1 elements, so k*(k+1)/2 differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The counting bound suggests max entry m must be at least roughly sqrt(2 * total_diffs)
    # But we want to minimize m. 
    # We will search for the minimum possible max entry (scope) that allows a valid solution.
    
    # Lower bound for max entry:
    # If max entry is M, the available differences are 1..M.
    # We need total_diffs distinct differences from 1..M.
    # So M >= total_diffs.
    # Wait, differences are a_j - a_i. If max entry is M, max diff is M.
    # So we need M >= total_diffs is a necessary condition? No.
    # The set of differences is a subset of {1, ..., M}.
    # So we need at least total_diffs available slots in {1, ..., M}.
    # Thus M >= total_diffs.
    
    # However, the structure of the rows imposes further constraints.
    # The "simple counting bound is 105" for (7,5) implies M >= 105.
    # total_diffs = 7 * (5*6/2) = 7 * 15 = 105.
    # So for (7,5), M >= 105.
    
    # We will try to find a solution with M = lower_bound, then M = lower_bound + 1, etc.
    # If we find one, we return it.
    # To improve score, we want the smallest M.
    
    lower_bound = total_diffs
    
    # We can also use a tighter lower bound based on the fact that each row sums to something?
    # No, simple counting is M >= total_diffs.
    
    # Let's define the CSP.
    # Variables: row_i for i in 0..n_rows-1.
    # Each row_i is a list of k+1 integers: 0 = x_0 < x_1 < ... < x_k <= M.
    # Constraints:
    # 1. All differences x_j - x_i (for i < j) across all rows must be distinct.
    
    def solve_csp(M):
        """
        Attempt to find n_rows valid rows with max entry <= M.
        Returns list of rows or None.
        """
        # We will build rows one by one.
        # To use MRV, we could pick the row with the fewest valid options first,
        # but since rows are symmetric, we can just fix an order and use backtracking.
        # Actually, symmetry breaking: we can assume row_0 is the "canonical" row?
        # No, rows are distinct sets of differences.
        
        # Let's use a recursive backtracking approach.
        # State: 
        #   used_diffs: set of all differences used so far.
        #   current_rows: list of rows built so far.
        
        # To make it efficient, we can pre-compute valid rows for a given M?
        # Number of rows with max <= M is huge.
        
        # Alternative: Backtrack on individual elements of the current row?
        # That might be too slow.
        
        # Let's try to generate candidate rows for a specific M.
        # A row is valid if its internal differences are distinct and not in used_diffs.
        
        # Heuristic: 
        # 1. Start with M = lower_bound.
        # 2. If no solution, increase M.
        
        # Since the search space is large, we need a strong pruning strategy.
        # Forward Checking: When placing an element in the current row, check if it conflicts with used_diffs.
        
        # We will implement a backtracking solver that builds rows one by one.
        # For each row, we try to find a valid set of elements.
        
        # To speed up row generation, we can use a greedy approach with backtracking within the row.
        
        def generate_row(used_diffs, M):
            """
            Generate a valid row of length k+1, starting with 0, max <= M,
            such that its differences are not in used_diffs.
            Returns a list of rows (candidates) or empty list.
            We will return the first few valid rows found to allow backtracking on row choice.
            """
            candidates = []
            # We need to find k+1 numbers: 0 = x0 < x1 < ... < xk <= M
            # such that all xj - xi are not in used_diffs.
            
            # Backtracking search for the row elements.
            def backtrack_row(current_row, last_x):
                if len(current_row) == k + 1:
                    # Check if all differences are valid (they should be by construction)
                    # Add to candidates
                    candidates.append(list(current_row))
                    return True # Stop finding more candidates for this row? 
                                # Or find multiple to allow variety?
                                # For efficiency, let's just find one valid row first.
                                # If this fails, the caller (row-level backtrack) will try another M or different row.
                                # Actually, we need to allow the *next* row to be different.
                                # So we should return a list of possible rows.
                                # But generating many is expensive.
                                # Let's just return the first valid row found.
                                # If the global search fails, we might need to try a different row for the same M.
                                # So we should implement a generator or return multiple.
                                # Let's return a list of up to 5 candidates.
                    if len(candidates) < 5:
                         return False # Continue searching for more candidates
                    else:
                         return True # Found enough
                 
                # Try next x
                for x in range(last_x + 1, M + 1):
                    # Check differences with all elements in current_row
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        current_row.append(x)
                        if backtrack_row(current_row, x):
                            return True
                        current_row.pop()
                return False
            
            backtrack_row([0], 0)
            return candidates
        
        # Global backtracking over rows
        def solve_rows(rows_built, used_diffs):
            if len(rows_built) == n_rows:
                return rows_built
            
            # MRV: Which row index has the fewest options?
            # Since rows are symmetric, we just build the next one.
            # But we can choose the "hardest" row? No, they are identical constraints.
            
            # Get candidate rows for the next slot
            candidates = generate_row(used_diffs, M)
            
            if not candidates:
                return None
            
            # Try each candidate
            for cand in candidates:
                new_used = set(used_diffs)
                # Add differences of cand
                for i in range(len(cand)):
                    for j in range(i+1, len(cand)):
                        new_used.add(cand[j] - cand[i])
                
                result = solve_rows(rows_built + [cand], new_used)
                if result is not None:
                    return result
            
            return None
        
        # Initial call
        return solve_rows([], set())

    # Search for the minimal M
    # Start from lower_bound
    # To handle cases where lower_bound is not achievable, increment M.
    # We can also add a randomization to the candidate generation to escape local minima.
    
    best_solution = None
    
    # We will try M from lower_bound upwards.
    # But we should also try random restarts for a fixed M if it's hard.
    
    # Optimization: If k is small, we can be more aggressive.
    # For (7,5), M starts at 105.
    
    current_M = lower_bound
    max_M_to_try = lower_bound + 20 # Don't go too far, we want the best score
    
    while current_M <= max_M_to_try and time.time() < deadline:
        # Try to solve for this M
        # Add some randomness to break symmetry or find different solutions
        random.seed(current_M) # Reproducible for specific M, but different for each M
        
        solution = solve_csp(current_M)
        
        if solution:
            best_solution = solution
            break # Found minimal M, stop
            
        current_M += 1
        
        # If we didn't find a solution, try again with different random seed?
        # The generate_row is deterministic for a given used_diffs.
        # To add randomness, we can shuffle the order of x in range(last_x+1, M+1)
        # or shuffle candidates.
        
    # If no solution found in the range, we might have missed one due to determinism.
    # Let's do a few random restarts for the best M we tried (or lower_bound).
    if not best_solution and time.time() < deadline - 10:
        # Try lower_bound with randomization
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(123 + attempt)
            # We need to modify solve_csp to accept a random seed or use global random
            # The current solve_csp is deterministic.
            # Let's create a randomized version.
            
            def solve_csp_random(M):
                def generate_row_random(used_diffs, M):
                    candidates = []
                    def backtrack_row(current_row, last_x):
                        if len(current_row) == k + 1:
                            candidates.append(list(current_row))
                            if len(candidates) >= 3:
                                return True
                            return False
                        # Shuffle the range to introduce randomness
                        next_vals = list(range(last_x + 1, M + 1))
                        random.shuffle(next_vals)
                        for x in next_vals:
                            valid = True
                            for a in current_row:
                                d = x - a
                                if d in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                current_row.append(x)
                                if backtrack_row(current_row, x):
                                    return True
                                current_row.pop()
                        return False
                    backtrack_row([0], 0)
                    return candidates
                
                def solve_rows(rows_built, used_diffs):
                    if len(rows_built) == n_rows:
                        return rows_built
                    candidates = generate_row_random(used_diffs, M)
                    if not candidates:
                        return None
                    # Shuffle candidates too
                    random.shuffle(candidates)
                    for cand in candidates:
                        new_used = set(used_diffs)
                        for i in range(len(cand)):
                            for j in range(i+1, len(cand)):
                                new_used.add(cand[j] - cand[i])
                        result = solve_rows(rows_built + [cand], new_used)
                        if result is not None:
                            return result
                    return None
                
                return solve_rows([], set())

            solution = solve_csp_random(lower_bound)
            if solution:
                best_solution = solution
                break
    
    if best_solution:
        return best_solution
    else:
        # Fallback: Return a valid but potentially suboptimal solution
        # Greedy construction
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
