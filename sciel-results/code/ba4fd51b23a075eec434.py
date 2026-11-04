"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    # Each row has (k+1) choose 2 differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry: must be at least total_diffs
    # Because differences are distinct integers in [1, M]
    lower_bound = total_diffs
    
    best_rows = None
    best_max = float('inf')
    
    def backtracking_search(seed, max_limit=None):
        """
        Build rows element by element using backtracking with forward checking.
        We fill positions (row, col) in row-major order.
        """
        rng = random.Random(seed)
        
        # State: 
        # rows: list of lists, partially filled
        # used_diffs: set of all differences used so far
        # 
        # We fill row by row. Within a row, we fill left to right.
        # When choosing a value for rows[r][c], it must be > rows[r][c-1] (if c>0)
        # and all differences with previous elements in the same row must be new.
        
        rows = [[0] + [None] * k for _ in range(n_rows)]
        used_diffs = set()
        
        # We'll use a recursive backtracking approach, but to avoid deep recursion issues
        # and to allow randomization, we'll implement it iteratively with a stack or 
        # use a simple recursive function with a depth limit.
        # Given n_rows=4, k=7, depth is 4*7=28, which is manageable.
        
        def solve(r, c):
            """
            Try to fill position (r, c).
            r: current row index (0 to n_rows-1)
            c: current column index (1 to k, since col 0 is always 0)
            """
            nonlocal used_diffs, rows, best_max, best_rows
            
            if time.time() > deadline:
                return False
            
            # If we've filled all rows, we're done
            if r == n_rows:
                # Calculate max entry
                current_max = max(max(row) for row in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [row[:] for row in rows]
                return True
            
            # If c == 0, this position is already 0, move to next
            if c == 0:
                return solve(r, 1)
            
            # Determine the minimum possible value for this position
            min_val = rows[r][c-1] + 1
            
            # Determine the maximum possible value
            if max_limit is not None:
                max_val = min(max_limit, best_max - 1)  # We want to improve on best_max
            else:
                max_val = best_max - 1 if best_max != float('inf') else 10000
            
            if min_val > max_val:
                return False
            
            # Candidate values: from min_val to max_val
            # To add randomness, we can shuffle the candidate order or pick a random subset
            # But we need to be careful: smaller values are generally better for minimizing max.
            # However, a slightly larger value now might allow smaller values later.
            
            # Strategy: Try candidates in a randomized order, biased towards smaller values.
            # Or: Try the smallest few candidates first, with some randomization.
            
            # Let's try a range of candidates. To avoid scanning too many, 
            # we can limit the number of candidates we try per node.
            # But for correctness, we should try all if we want to find the optimal.
            # Given time limits, we'll try a subset.
            
            # Generate candidates
            # We'll try all values from min_val to max_val, but in a randomized order 
            # with a bias towards smaller values.
            
            # Actually, a better heuristic: try the smallest valid candidate first, 
            # but with some probability, try a larger one to escape local minima.
            
            # Let's try a limited set of candidates: the first 'width' valid candidates.
            width = 5  # Try up to 5 candidates per node
            
            candidates = []
            for x in range(min_val, max_val + 1):
                # Check if x is valid for this position
                valid = True
                new_diffs = set()
                for prev in rows[r][:c]:
                    d = x - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    candidates.append((x, new_diffs))
                    if len(candidates) >= width:
                        break
            
            if not candidates:
                return False
            
            # Randomize the order of candidates, but keep a bias towards smaller x
            # We can shuffle, or pick a random one from the first few.
            # Let's shuffle the candidates list.
            rng.shuffle(candidates)
            
            for x, new_diffs in candidates:
                # Try this candidate
                rows[r][c] = x
                used_diffs.update(new_diffs)
                
                # Recurse
                if c < k:
                    if solve(r, c + 1):
                        return True
                else:
                    if solve(r + 1, 0):
                        return True
                
                # Backtrack
                used_diffs.difference_update(new_diffs)
                rows[r][c] = None
            
            return False
        
        # Start solving
        success = solve(0, 0)
        return success
    
    # Initial greedy baseline
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
    
    # Try backtracking search with different seeds and max limits
    # We want to find a solution with max < best_max
    # Start with a limit slightly below current best, and increase if needed
    
    current_seed = 0
    # We'll try to find a solution with max < best_max
    # If we fail, we can try a higher limit, but we only care about improving.
    
    while time.time() < deadline:
        current_seed += 1
        
        # Try to find a solution with max < best_max
        # The max limit for the search is best_max - 1
        # But we also need to ensure that the search space is not too large.
        
        # Call backtracking search
        # Note: The backtracking search modifies best_max and best_rows if it finds a better solution.
        # But it also uses 'rows' and 'used_diffs' which are local to the function.
        # The 'best_max' and 'best_rows' are nonlocal, so they are updated correctly.
        
        # However, the backtracking search might be slow if best_max is large.
        # Let's add a timeout inside the backtracking search, which we already have.
        
        backtracking_search(current_seed, max_limit=best_max - 1)
        
        # If we found a solution, best_max is updated.
        # We can stop if we hit the lower bound.
        if best_max == lower_bound:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
