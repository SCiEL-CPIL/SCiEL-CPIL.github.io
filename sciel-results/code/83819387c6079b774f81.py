import sys
import time

def construct(n_rows, k):
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound: we need total_diffs distinct positive differences,
    # each at most M, so M >= total_diffs
    lower_bound = total_diffs
    
    # For (7,5): total_diffs = 7*5*6//2 = 105
    # Best known is 112, so search from lower_bound upward
    
    # Use a time limit to avoid infinite loops
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    for M in range(lower_bound, lower_bound + 30):
        if time.time() - start_time > time_limit:
            break
        result = solve_csp(n_rows, k, M, time_limit - (time.time() - start_time))
        if result is not None:
            return result
    
    # Fallback to greedy if backtracking fails or times out
    return greedy_fallback(n_rows, k)

def solve_csp(n_rows, k, M, time_budget):
    """
    Solve the difference triangle problem using CSP with bitset-based backtracking.
    Rows are selected in order of increasing difficulty (fewest valid extensions).
    """
    start_time = time.time()
    
    # State:
    # used_diffs: integer bitset where bit d is set if difference d is used
    # rows: list of completed rows
    # available_rows: which rows are still to be assigned
    
    used_diffs = 0
    rows = [None] * n_rows
    unassigned = list(range(n_rows))
    
    # Precompute: for each possible M, what's the max difference
    # We'll use recursive backtracking with difficulty ordering
    
    def count_extensions(used_diffs, M, k):
        """Count how many valid rows can be formed with current used_diffs."""
        count = 0
        row = [0]
        
        def dfs(index):
            nonlocal count
            if index == k + 1:
                count += 1
                return
            last_val = row[-1]
            # Try next values
            for x in range(last_val + 1, M + 1):
                # Check if all differences x - a_i are available
                valid = True
                for a in row:
                    diff = x - a
                    if (used_diffs >> diff) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    dfs(index + 1)
                    row.pop()
        
        dfs(0)
        return count
    
    def get_row_diffs(row):
        """Get the set of differences in a row as a bitset."""
        bits = 0
        for i in range(len(row)):
            for j in range(i):
                diff = row[i] - row[j]
                bits |= (1 << diff)
        return bits
    
    def backtrack(used_diffs, rows, unassigned, depth):
        if time.time() - start_time > time_budget:
            return False, "timeout"
        
        if not unassigned:
            return True, None
        
        # Select the row with fewest valid extensions (difficulty ordering)
        best_row = -1
        best_count = float('inf')
        best_row_data = None
        
        for row_idx in unassigned:
            # Count valid rows for this position
            count, row_data = count_valid_rows(used_diffs, M, k)
            if count == 0:
                return False, "dead_end"
            if count < best_count:
                best_count = count
                best_row = row_idx
                best_row_data = row_data
        
        if best_row == -1:
            return False, "no_rows"
        
        # Try each valid row for the most constrained position
        for row_data in best_row_data:
            row_bits = get_row_diffs(row_data)
            if (used_diffs & row_bits) != 0:
                continue
            
            new_used = used_diffs | row_bits
            rows[best_row] = row_data
            new_unassigned = [r for r in unassigned if r != best_row]
            
            result, err = backtrack(new_used, rows, new_unassigned, depth + 1)
            if result:
                return True, None
            
            rows[best_row] = None
        
        return False, "exhausted"
    
    def count_valid_rows(used_diffs, M, k):
        """
        Count and enumerate valid rows given current used_diffs.
        Returns (count, list_of_rows)
        """
        valid_rows = []
        row = [0]
        
        def dfs(index):
            if index == k + 1:
                valid_rows.append(list(row))
                return
            last_val = row[-1]
            # Limit search: we need at least (k+1-index) more elements
            # Each needs at least 1 more than previous
            min_remaining = k + 1 - index
            max_x = M
            for x in range(last_val + 1, max_x + 1):
                # Check differences
                valid = True
                for a in row:
                    diff = x - a
                    if (used_diffs >> diff) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    dfs(index + 1)
                    row.pop()
        
        dfs(0)
        return len(valid_rows), valid_rows
    
    success, err = backtrack(used_diffs, rows, unassigned, 0)
    
    if success:
        # Verify all rows are filled
        if all(r is not None for r in rows):
            return rows
    
    return None

def greedy_fallback(n_rows, k):
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
