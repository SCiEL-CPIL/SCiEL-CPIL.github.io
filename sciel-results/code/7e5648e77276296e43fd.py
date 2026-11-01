import sys

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set.
    n_rows: number of rows
    k: parameter such that row size is k + 1
    """
    size = k + 1
    
    # Calculate the minimum possible maximum element based on counting bound
    # Total distinct differences required = n_rows * C(size, 2)
    # These differences must be in [1, M]
    total_diffs_needed = n_rows * (size * (size - 1) // 2)
    M_lower = total_diffs_needed
    
    def get_valid_rows(M, size):
        """
        Generate all valid rows of given size in [0, M] starting with 0.
        A row is valid if all positive differences within it are distinct.
        """
        rows = []
        curr = [0]
        curr_diffs = set()
        
        def backtrack(last_val, curr_row, curr_d):
            if len(curr_row) == size:
                rows.append(list(curr_row))
                return
            
            # Pruning: not enough numbers left to fill the row
            if M - last_val < size - len(curr_row):
                return
                
            for x in range(last_val + 1, M + 1):
                # Check if x - a is in curr_d for any a in curr_row
                valid = True
                new_diffs = []
                for a in curr_row:
                    d = x - a
                    if d in curr_d:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    curr_row.append(x)
                    curr_d.update(new_diffs)
                    backtrack(x, curr_row, curr_d)
                    curr_row.pop()
                    for d in new_diffs:
                        curr_d.remove(d)
                        
        backtrack(0, curr, curr_diffs)
        return rows

    # Search for the minimal M
    # We check a range of M values starting from the lower bound
    # If no solution is found, we expand the range
    search_range = 30 
    
    for M in range(M_lower, M_lower + search_range):
        valid_rows = get_valid_rows(M, size)
        
        if not valid_rows:
            continue
            
        # Prepare data for search
        row_data = []
        for r in valid_rows:
            diffs = set()
            for i in range(size):
                for j in range(i + 1, size):
                    diffs.add(r[j] - r[i])
            row_data.append((r[-1], diffs, r))
            
        # Sort by max element to potentially find solutions with smaller max elements earlier
        # though we are iterating M, this helps if multiple solutions exist
        row_data.sort(key=lambda x: x[0])
        
        # Backtracking to find n_rows rows with disjoint difference sets
        def find_solution(idx, selected, used_diffs):
            if len(selected) == n_rows:
                return selected
            
            if idx >= len(row_data):
                return None
            
            # Pruning: not enough rows left
            if len(row_data) - idx < n_rows - len(selected):
                return None
                
            # Option 1: Skip current row
            res = find_solution(idx + 1, selected, used_diffs)
            if res:
                return res
            
            # Option 2: Take current row
            max_val, diffs, row = row_data[idx]
            if not (diffs & used_diffs):
                new_used = used_diffs | diffs
                res = find_solution(idx + 1, selected + [row], new_used)
                if res:
                    return res
            
            return None

        solution = find_solution(0, [], set())
        
        if solution:
            return solution

    # Fallback: Greedy approach if search fails or is too slow
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < size:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
