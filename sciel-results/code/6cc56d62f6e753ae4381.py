import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # 1. Generate initial greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    rows, used = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Lower bound estimation
    # Total diffs = n_rows * k(k+1)/2
    # Max diff <= M
    # Count of distinct diffs available in [1, M] is M
    # So n_rows * k(k+1)/2 <= M
    # M >= n_rows * k(k+1) / 2
    lower_bound = int((n_rows * k * (k + 1)) / 2) + 1
    
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(k+1) for j in range(i+1, k+1)}

    def find_row(max_val, forbidden_diffs):
        """
        Try to find a row of length k+1, starting at 0, strictly increasing,
        max <= max_val, such that its diffs are disjoint from forbidden_diffs.
        Returns the row if found, else None.
        """
        row = [0]
        row_diffs = set()
        
        def backtrack():
            if len(row) == k + 1:
                return True
            
            last = row[-1]
            # We need to pick next element x > last
            # x can be at most max_val
            # Optimization: iterate x from last+1 to max_val
            # To improve chance of finding a compact row, maybe iterate from last+1 upwards?
            # Yes, smaller x is better for minimizing max.
            
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                # Check conflict with row's own diffs
                if new_diffs & row_diffs:
                    continue
                # Check conflict with forbidden diffs
                if new_diffs & forbidden_diffs:
                    continue
                
                # Accept x
                row.append(x)
                row_diffs |= new_diffs
                
                if backtrack():
                    return True
                
                # Backtrack
                row.pop()
                row_diffs -= new_diffs
                
            return False
            
        if backtrack():
            return row
        return None

    # Local Search Loop
    # We will try to improve the solution by replacing rows with smaller ones.
    # We keep the current state in `rows` and `used`.
    # `rows` is mutable.
    
    # To avoid getting stuck in a very local minimum, we can occasionally restart
    # or shuffle the order of rows we try to improve.
    
    rows_to_try = list(range(n_rows))
    
    while time.time() < deadline:
        random.shuffle(rows_to_try)
        improved = False
        
        for idx in rows_to_try:
            if time.time() > deadline:
                break
                
            current_row = rows[idx]
            current_max = current_row[-1]
            
            # If this row is already at the lower bound, skip
            if current_max <= lower_bound:
                continue
            
            # Calculate forbidden diffs (all diffs except those in this row)
            current_diffs = get_row_diffs(current_row)
            forbidden_diffs = used - current_diffs
            
            # Try to find a row with smaller max
            # We try target_max from current_max - 1 down to lower_bound
            # Limit the search depth if it takes too long? 
            # The backtracking is exponential in k. For k=7, depth is 7.
            # Branching factor is roughly max_val - last.
            # It should be fast enough for small improvements.
            
            new_row = None
            # Try to find a row that is strictly smaller
            # To speed up, we can try a few random target_max values or just go down.
            # Going down is systematic.
            
            # Optimization: If current_max is very large, we might want to jump down a bit?
            # No, greedy reduction is safer.
            
            for target_max in range(current_max - 1, lower_bound - 1, -1):
                # Time check inside the loop? 
                # Backtracking is fast, but if it fails, it might take a while.
                # We'll rely on the outer loop time check.
                
                candidate = find_row(target_max, forbidden_diffs)
                if candidate:
                    new_row = candidate
                    break
            
            if new_row:
                # Update state
                rows[idx] = new_row
                new_diffs = get_row_diffs(new_row)
                used = (used - current_diffs) | new_diffs
                
                new_global_max = max(max(r) for r in rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [r[:] for r in rows]
                    improved = True
                    
                # If we improved, we might want to continue trying to improve other rows
                # or restart the shuffle?
                # Continuing is fine.
                
        if not improved:
            # If no improvement in a full pass, we might be stuck.
            # We can try a "kick" by randomizing a row's structure?
            # Or just accept that we might have found a local optimum.
            # Given the time limit, we just keep shuffling and trying.
            # Sometimes a different order of rows helps.
            pass

    return best_rows
