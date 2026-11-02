import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Lower bound for M: A Golomb ruler of length L has at least L(L-1)/2 differences.
    # Total differences needed: n_rows * (k+1)*k/2
    # Total differences available in [0, M]: M*(M+1)/2
    # So M(M+1)/2 >= n_rows * (k+1)*k/2
    # M^2 >= n_rows * k * (k+1)
    # M >= sqrt(n_rows * k * (k+1))
    
    total_diffs_needed = n_rows * (k + 1) * k // 2
    lower_bound = 0
    # Simple quadratic approximation
    import math
    lower_bound = int(math.ceil(math.sqrt(n_rows * k * (k + 1))))
    # Add a small buffer
    lower_bound = max(lower_bound, k * (k + 1) // 2) # At least one ruler length
    
    # Upper bound from a simple greedy run
    def greedy_construct(n, k):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, max(max(r) for r in rows)

    greedy_rows, greedy_max = greedy_construct(n_rows, k)
    upper_bound = greedy_max
    
    best_solution = greedy_rows
    best_max = greedy_max
    
    # Try to find a better solution with M < greedy_max
    # Iterate M from lower_bound to greedy_max - 1
    # If found, return it.
    
    def dfs(row_idx, current_row, used_diffs, M):
        """
        Recursively build rows.
        row_idx: index of the row we are currently building (0 to n_rows-1)
        current_row: list of integers in the current row
        used_diffs: set of all differences used in completed rows and current row so far
        M: maximum allowed value
        Returns: list of rows if successful, None if failed
        """
        if time.time() > deadline:
            return None
            
        if row_idx == n_rows:
            # All rows built successfully
            return [current_row] # This is a bit tricky, we need to accumulate rows
            
        # This structure is awkward for returning the full list.
        # Let's use a mutable list for rows.
        pass

    # Better DFS structure
    rows = []
    used_diffs = set()
    
    def backtrack(row_idx, current_row, M):
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            # Move to next row
            if backtrack(row_idx + 1, [0], M):
                return True
            # Backtrack: remove the row
            rows.pop()
            return False
            
        # Try to extend current_row
        last_val = current_row[-1]
        # Start from last_val + 1 to M
        # Optimization: We can skip values that create duplicate diffs within the current row
        # or with used_diffs.
        
        for x in range(last_val + 1, M + 1):
            # Check if x can be added to current_row
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs or d in [nd for nd in new_diffs]:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Tentatively add x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, M):
                    return True
                    
                # Undo
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
        return False

    # We need to be careful with the `used_diffs` set modification.
    # `difference_update` is O(len(new_diffs)), which is fine.
    
    # Let's try M values
    for M in range(lower_bound, greedy_max):
        # Reset state
        rows = []
        used_diffs = set()
        
        # Initial call
        if backtrack(0, [0], M):
            # Found a solution with max M
            # The rows list contains the solution
            return rows
            
    # If no better solution found, return greedy
    return greedy_rows

# EVOLVE-BLOCK-END
