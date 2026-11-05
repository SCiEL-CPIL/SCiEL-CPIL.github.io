import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    # Each row has k+1 elements, so C(k+1,2) differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry: we need total_diffs distinct positive integers
    # The smallest possible max entry m must satisfy m*(m+1)/2 >= total_diffs
    # So m >= ~sqrt(2*total_diffs)
    import math
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # Try to find a solution with max entry as small as possible
    # Start from lower bound and increase
    
    best_solution = None
    best_max = float('inf')
    
    def compute_row_differences(row):
        """Compute all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for d in compute_row_differences(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True
    
    def max_entry(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def greedy_construct(max_val, n_rows, k, seed=None):
        """
        Greedy construction with a cap on max entry.
        Each row tries to use values in [0, max_val].
        Returns a valid set of rows or None if it fails.
        """
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            x = 1
            success = True
            while len(row) < k + 1:
                # Try next value
                found = False
                while x <= max_val:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    success = False
                    break
            if not success:
                return None
            rows.append(row)
        
        return rows
    
    # First, try greedy from the lower bound
    for trial_max in range(lower_bound, lower_bound + 50):
        if time.time() > deadline:
            break
        sol = greedy_construct(trial_max, n_rows, k)
        if sol is not None:
            best_solution = sol
            best_max = trial_max
            break
    
    # If greedy didn't find anything close to optimal, do local search
    if best_solution is None:
        # Fallback: use a larger max value for greedy
        sol = greedy_construct(lower_bound + 200, n_rows, k)
        if sol is not None:
            best_solution = sol
            best_max = max_entry(sol)
    elif time.time() < deadline - 5.0:
        # Try to improve via local search: rebuild rows with randomization
        # Strategy: keep the set of used differences, but rebuild rows in random order
        # or with randomized choices where possible
        
        def local_search_improve(rows, max_val, n_rows, k, time_limit=30):
            """
            Try to improve by rebuilding rows one at a time with different strategies.
            """
            start = time.time()
            current = [list(r) for r in rows]
            current_max = max_entry(current)
            
            used_diffs = set()
            for row in current:
                used_diffs |= compute_row_differences(row)
            
            for iteration in range(500):
                if time.time() - start > time_limit:
                    break
                if time.time() > deadline:
                    break
                
                # Pick a random row to rebuild
                r = random.randint(0, n_rows - 1)
                
                # Remove its differences from used set
                row_diffs = compute_row_differences(current[r])
                used_diffs -= row_diffs
                
                # Try to rebuild this row with a different strategy
                # Strategy 1: Greedy with random tie-breaking
                row = [0]
                x = 1
                success = True
                while len(row) < k + 1:
                    candidates = []
                    while x <= current_max:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                        x += 1
                    if not candidates:
                        success = False
                        break
                    # Pick randomly from candidates (or the first few for efficiency)
                    pick_idx = random.randint(0, min(5, len(candidates) - 1))
                    chosen = candidates[pick_idx]
                    row.append(chosen)
                    used_diffs |= {chosen - a for a in row[:-1]}
                    x = chosen + 1
                
                if success and max(row) <= current_max:
                    current[r] = row
                    # Check if we can reduce max_val
                    new_max = max_entry(current)
                    if new_max < current_max:
                        current_max = new_max
                        used_diffs = set()
                        for row in current:
                            used_diffs |= compute_row_differences(row)
            
            return current, current_max
        
        improved, improved_max = local_search_improve(best_solution, best_max, n_rows, k)
        if improved_max < best_max:
            best_solution = improved
            best_max = improved_max
    
    if best_solution is None:
        # Last resort: very generous greedy
        sol = greedy_construct(1000, n_rows, k)
        if sol is not None:
            return sol
        else:
            # Should never happen for reasonable inputs
            return [[0]] * n_rows
    
    return best_solution
