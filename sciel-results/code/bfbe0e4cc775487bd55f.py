"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking search with pruning and symmetry breaking.
    """
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on the maximum entry M:
    # We need at least total_diffs distinct positive differences.
    # The differences must be in [1, M]. So M >= total_diffs.
    # Also, each row has k+1 entries, so the max entry in a row is at least k (trivial).
    # A better bound: for a single row with k+1 entries from 0 to M, the max difference is M.
    # But we need all differences across all rows to be distinct.
    
    # Let's use a lower bound based on the fact that differences are distinct.
    # The smallest possible set of total_diffs distinct positive integers is 1, 2, ..., total_diffs.
    # So M must be at least total_diffs. But also, within a single row, the differences 
    # between consecutive elements sum to M (the max entry). 
    # Actually, a tighter bound: consider that in each row, the k+1 elements a_0=0 < a_1 < ... < a_k=M_row.
    # The differences a_j - a_i for i<j must be distinct across all rows.
    # The number of differences per row is k*(k+1)/2.
    # The maximum possible difference in a row is M_row.
    # So all differences must be <= max(M_row) over all rows.
    # We need total_diffs distinct values in [1, M_max].
    # So M_max >= total_diffs.
    
    # Let's start searching from a reasonable lower bound.
    # For the specific case (7,5): total_diffs = 7 * 15 = 105.
    # The counting bound is 105, best known is 112.
    
    # We'll use a backtracking approach with the following strategy:
    # 1. Fix the first row to be as "compact" as possible to save space for others.
    # 2. Use symmetry breaking: sort rows by their max entry.
    # 3. Prune when the remaining differences can't fit in the remaining range.
    
    # Actually, let's try a different approach: use the known structure.
    # For small n_rows and k, we can use a smart search.
    
    # Let's define a helper to check if a partial assignment is valid.
    
    # For now, let's implement a backtracking search with good pruning.
    
    # The key insight: we need to assign n_rows * (k+1) numbers (with first of each row = 0)
    # such that all pairwise differences within each row are globally distinct.
    
    # Let's represent a row by its k non-zero elements (since the first is always 0).
    # A row is: [0, x_1, x_2, ..., x_k] with 0 < x_1 < x_2 < ... < x_k.
    
    # The differences are: x_i - 0 = x_i, and x_j - x_i for 1 <= i < j <= k.
    # So the set of differences for a row is: {x_1, x_2, ..., x_k} union {x_j - x_i : 1 <= i < j <= k}.
    
    # We need all these differences across all rows to be distinct.
    
    # Strategy: 
    # - We'll search for rows one by one.
    # - For each row, we try to construct it greedily but with backtracking.
    # - We maintain a set of used differences.
    # - We also track the current maximum entry to minimize it.
    
    # To make this efficient, we'll use the following:
    # 1. Order rows: we'll try to make early rows have smaller max entries.
    # 2. For each row, we generate candidate rows in order of their max entry.
    
    # Given the constraints (120s timeout), let's be practical.
    # For (7,5), the answer is 112. Let's see if we can find it.
    
    # Let's implement a backtracking search with the following:
    # - State: list of completed rows, set of used differences, current row being built.
    # - We'll use a priority queue or sorted list of candidate next values.
    
    # Actually, let's try a simpler approach first: 
    # Use the greedy approach but with a twist: try multiple starting points and take the best.
    
    # Or, we can use a known result: for (7,5), the optimal is 112.
    # Let's hardcode a good heuristic: start with the greedy approach, but if we get stuck,
    # backtrack and try different choices.
    
    # Let's implement a proper backtracking search.
    
    def solve():
        used_diffs = set()
        rows = []
        max_entry = 0
        
        def backtrack(row_idx, current_row, current_used, current_max):
            nonlocal max_entry
            
            if row_idx == n_rows:
                max_entry = min(max_entry, current_max)
                return True
            
            # Pruning: if current_max already exceeds the best known, prune.
            # Also, we need to place (n_rows - row_idx) more rows.
            # Each row needs k+1 entries, with the first being 0.
            # The remaining rows must have max entries that allow their differences to fit.
            
            # Simple pruning: the minimum possible max for the remaining rows
            # is at least the current_max (since we're minimizing the global max).
            # Actually, we want to minimize the global max, so if current_max is already
            # high, we might want to try to keep it low.
            
            # Let's just try to build the next row.
            # We'll try candidate values for the next element in the current row.
            
            if len(current_row) == k + 1:
                # Row is complete. Check if it's valid (differences not in used_diffs).
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in used_diffs or d in row_diffs:
                            return False
                        row_diffs.add(d)
                
                # Add this row and recurse.
                new_used = used_diffs | row_diffs
                new_max = max(current_max, current_row[-1])
                
                if backtrack(row_idx + 1, [0], new_used, new_max):
                    rows.append(current_row)
                    return True
                return False
            
            # Determine the range of candidate values for the next element.
            last_val = current_row[-1] if current_row else 0
            # The next value must be > last_val.
            # Upper bound: we don't want to exceed current_max if we're trying to keep it low,
            # but we might need to go higher.
            # Let's set a reasonable upper bound.
            
            # The maximum difference we can have is bounded by the global max.
            # Let's try values from last_val + 1 up to some limit.
            
            # To be efficient, let's try a limited range.
            # The next value x must satisfy: x - a_i for all a_i in current_row are not in used_diffs.
            
            # Let's try values in increasing order.
            # Upper bound for x: we need x - last_val to be a valid difference.
            # Also, x should not be too large.
            
            # Let's set an upper bound based on the remaining capacity.
            # We have total_diffs differences needed. The differences used so far are len(used_diffs).
            # The remaining differences must be in [1, M] where M is the global max.
            # So M >= len(used_diffs) + remaining_diffs_needed.
            
            remaining_rows = n_rows - row_idx
            diffs_in_current_row = len(current_row) * (len(current_row) - 1) // 2
            remaining_diffs = total_diffs - len(used_diffs) - diffs_in_current_row
            
            # The global max must be at least remaining_diffs + len(used_diffs) = total_diffs.
            # But also, the current row's max must be at least its last element.
            
            # Let's just try a reasonable range.
            # For the next element, try from last_val + 1 to last_val + k * 20 or so.
            
            # Actually, let's be smarter: the next value x must be such that
            # x - a_i is not in used_diffs for all a_i in current_row.
            
            # Let's try values in a window.
            start = last_val + 1
            # Upper bound: let's try up to start + 50 or so.
            # This is a heuristic; we might need to go higher.
            
            # Let's try a wider range but with early termination.
            for x in range(start, start + 100):
                # Check if x - a_i is not in used_diffs for all a_i in current_row.
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check that x - a_i are distinct within the current row's new differences.
                # Actually, since current_row is strictly increasing, x - a_i are all distinct
                # and positive. And we already checked they're not in used_diffs.
                
                # Try this value.
                new_current_row = current_row + [x]
                new_used = used_diffs | {x - a for a in current_row}
                
                # Recurse.
                if backtrack(row_idx, new_current_row, new_used, max(current_max, x)):
                    return True
            
            return False
        
        # Start the search.
        # We need to find the minimum max_entry.
        # Let's do a binary search on the max_entry, or just try to find a solution
        # and then try to improve it.
        
        # For now, let's just find any valid solution, then try to minimize.
        # Actually, let's just run the backtracking and see what max we get.
        
        # To speed things up, let's add a timeout-like mechanism by limiting the search.
        # But for now, let's just run it.
        
        # Initialize.
        initial_row = [0]
        initial_used = set()
        initial_max = 0
        
        # We'll store the best solution found.
        best_solution = None
        best_max = float('inf')
        
        # Let's modify backtrack to store the best solution.
        
        def backtrack_best(row_idx, current_row, current_used, current_max):
            nonlocal best_solution, best_max
            
            # Prune if current_max >= best_max.
            if current_max >= best_max:
                return
            
            if row_idx == n_rows:
                # Found a solution.
                if current_max < best_max:
                    best_max = current_max
                    best_solution = [list(r) for r in rows] + [list(current_row)]
                    # Actually, we need to store all rows. Let's fix this.
                return
            
            # If current row is complete, move to next row.
            if len(current_row) == k + 1:
                # Check validity.
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in current_used or d in row_diffs:
                            return
                        row_diffs.add(d)
                
                new_used = current_used | row_diffs
                new_max = max(current_max, current_row[-1])
                
                # Save current row.
                rows.append(current_row)
                if backtrack_best(row_idx + 1, [0], new_used, new_max):
                    pass
                rows.pop()
                return
            
            # Try next value.
            last_val = current_row[-1]
            for x in range(last_val + 1, last_val + 100):
                valid = True
                for a in current_row:
                    d = x - a
                    if d in current_used:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_current_row = current_row + [x]
                new_used = current_used | {x - a for a in current_row}
                
                backtrack_best(row_idx, new_current_row, new_used, max(current_max, x))
                # If we found a good solution, we might want to stop.
                # But let's continue to find better solutions.
        
        # This approach is getting complicated. Let's simplify.
        
        # Let's just use the greedy approach with a few improvements:
        # 1. Try multiple random orderings of the rows.
        # 2. Use a better heuristic for choosing the next value.
        
        # Actually, for the specific task (7,5), let's just use a known good set.
        # But we need a general solution.
        
        # Let's implement a simple but effective backtracking with a time limit.
        
        import time
        start_time = time.time()
        timeout = 100  # seconds
        
        # We'll use a global best.
        global_best = None
        global_best_max = float('inf')
        
        def bt(row_idx, current_row, used, max_val):
            nonlocal global_best, global_best_max
            
            if time.time() - start_time > timeout:
                return False
            
            if max_val >= global_best_max:
                return False
            
            if row_idx == n_rows:
                global_best_max = max_val
                global_best = [list(r) for r in rows] + [list(current_row)]
                return True
            
            if len(current_row) == k + 1:
                # Check if the row is valid.
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in used or d in row_diffs:
                            return False
                        row_diffs.add(d)
                
                rows.append(current_row)
                result = bt(row_idx + 1, [0], used | row_diffs, max(max_val, current_row[-1]))
                rows.pop()
                return result
            
            last = current_row[-1]
            # Try values in a limited range.
            for x in range(last + 1, last + 50):
                ok = True
                for a in current_row:
                    if (x - a) in used:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_row = current_row + [x]
                new_used = used | {x - a for a in current_row}
                
                if bt(row_idx, new_row, new_used, max(max_val, x)):
                    return True
            
            return False
        
        # Start with an empty row.
        rows = []
        bt(0, [0], set(), 0)
        
        if global_best is not None:
            return global_best
        
        # Fallback: use the greedy approach.
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
    
    return solve()
# EVOLVE-BLOCK-END
