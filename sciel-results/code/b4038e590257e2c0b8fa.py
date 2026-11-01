import sys

sys.setrecursionlimit(10000)

def greedy_bound(n_rows, k):
    """
    Runs the simple greedy algorithm to find an upper bound for the max value.
    """
    used = set()
    max_val = 0
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                max_val = max(max_val, x)
    return max_val

def construct(n_rows, k):
    K = k * (k + 1) // 2
    # Lower bound: total differences must fit in [1, L]
    low = n_rows * K
    
    # Get an upper bound using greedy
    ub = greedy_bound(n_rows, k)
    
    # Helper function for DFS
    def find_solution(L):
        """
        Tries to construct n_rows rows with max value <= L.
        Returns a list of rows if successful, else None.
        """
        rows = []
        
        def dfs(row_idx, used_mask, current_row):
            # Pruning: Check capacity
            m = len(current_row)
            # Diffs generated so far in current row
            current_diffs_count = m * (m - 1) // 2
            # Diffs needed to complete current row
            rem_current = K - current_diffs_count
            # Diffs needed for all future rows
            rem_future = (n_rows - row_idx - 1) * K
            total_needed = rem_current + rem_future
            
            used_count = used_mask.bit_count()
            # Available slots in [1, L]
            available = L - used_count
            
            if available < total_needed:
                return None
            
            if row_idx == n_rows:
                # All rows constructed
                return rows[:]
            
            if len(current_row) == k + 1:
                # Current row complete
                rows.append(current_row)
                result = dfs(row_idx + 1, used_mask, [0])
                if result:
                    return result
                rows.pop()
                return None
            
            last_val = current_row[-1]
            
            # Try next values
            for x in range(last_val + 1, L + 1):
                # Check differences
                valid = True
                new_mask = used_mask
                for a in current_row:
                    d = x - a
                    # d must be > 0
                    # Check if d is already used
                    if (new_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if not valid:
                    continue
                
                # Recurse
                result = dfs(row_idx, new_mask, current_row + [x])
                if result:
                    return result
            
            return None
        
        return dfs(0, 0, [0])

    # Linear search from low to ub
    for L in range(low, ub + 1):
        sol = find_solution(L)
        if sol:
            return sol
    
    # If no solution found within [low, ub], return the greedy solution
    return greedy_construct(n_rows, k)

def greedy_construct(n_rows, k):
    """
    Fallback greedy construction.
    """
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
