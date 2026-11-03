import heapq
import time

def construct(n_rows, k):
    """
    Construct difference triangle set using A* search on bitmask state space.
    State: (used_diff_mask, completed_rows_count, path)
    Transition: add a valid row configuration
    Heuristic: lower bound on scope from remaining difference slots
    """
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Precompute binomial coefficients
    def binom(n, r):
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result
    
    # Number of differences per row: C(k+1, 2)
    diffs_per_row = binom(k + 1, 2)
    total_diffs_needed = n_rows * diffs_per_row
    
    # Admissible heuristic: given used_mask and rows_done,
    # find the smallest scope M such that there are at least
    # (total_diffs_needed - used_count) available differences in [1, M]
    def get_heuristic(used_mask, rows_done):
        # Count how many differences are already used
        used_count = bin(used_mask).count('1')
        remaining = total_diffs_needed - used_count
        
        if remaining <= 0:
            return 0
        
        # Find the smallest M such that number of available diffs in [1, M] >= remaining
        # Available diffs in [1, M] = M - (number of used diffs <= M)
        # We need to find smallest M where: M - used_leq_M >= remaining
        
        # Binary search on M
        lo = 1
        hi = max(used_mask.bit_length(), remaining + 10)
        
        while lo < hi:
            mid = (lo + hi) // 2
            # Count used diffs <= mid
            mask_leq_mid = (1 << (mid + 1)) - 1
            used_leq_mid = bin(used_mask & mask_leq_mid).count('1')
            available = mid - used_leq_mid
            if available >= remaining:
                hi = mid
            else:
                lo = mid + 1
        
        return lo
    
    # Generate all valid row configurations for a given used_mask up to max_val
    def generate_rows(used_mask, max_val):
        """Generate all valid rows up to max_val, yield (row_tuple, diff_mask)"""
        rows = []
        
        def backtrack(row, start, diff_mask):
            if len(row) == k + 1:
                rows.append((tuple(row), diff_mask))
                return
            
            for x in range(start, max_val + 1):
                valid = True
                new_mask = diff_mask
                for a in row:
                    d = x - a
                    if d > max_val:
                        break
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    row.append(x)
                    backtrack(row, x + 1, new_mask)
                    row.pop()
        
        backtrack([0], 1, 0)
        return rows
    
    # A* search
    # State: (f, h, g, used_mask, rows_done, path)
    # g = current scope (max element in all rows so far)
    # f = max(g, h)
    # path = list of completed rows
    
    # Start state: no rows done, no diffs used, scope = 0, empty path
    start_h = get_heuristic(0, 0)
    start_state = (start_h, start_h, 0, 0, 0, ())  # f, h, g, used_mask, rows_done, path
    
    open_list = [start_state]
    visited = {}  # (used_mask, rows_done) -> best g seen
    
    best_found = None
    best_scope = float('inf')
    
    while open_list:
        if time.time() - start_time > timeout:
            break
        
        f, h, g, used_mask, rows_done, path = heapq.heappop(open_list)
        
        state_key = (used_mask, rows_done)
        
        # If we've seen this state with a better or equal g, skip
        if state_key in visited and visited[state_key] <= g:
            continue
        visited[state_key] = g
        
        # If we've completed all rows
        if rows_done == n_rows:
            best_found = list(path)
            best_scope = g
            return best_found
        
        # Expand: try to add a new row
        # Upper bound for new row's max: we can't exceed current best_scope
        max_row_val = min(best_scope if best_scope < float('inf') else g + 15, 150)
        
        # Generate candidate rows
        candidate_rows = generate_rows(used_mask, max_row_val)
        
        # For each candidate, compute new state
        candidates = []
        for row_tuple, new_diff_mask in candidate_rows:
            new_used = used_mask | new_diff_mask
            new_rows_done = rows_done + 1
            new_g = max(g, row_tuple[-1])  # scope is max of all row maxes
            new_path = path + (row_tuple,)
            
            new_h = get_heuristic(new_used, new_rows_done)
            new_f = max(new_g, new_h)
            
            # Prune if this state was already visited with better g
            child_key = (new_used, new_rows_done)
            if child_key in visited and visited[child_key] <= new_g:
                continue
            
            candidates.append((new_f, new_h, new_g, new_used, new_rows_done, new_path))
        
        # Push all candidates
        for state in candidates:
            heapq.heappush(open_list, state)
    
    # If A* didn't find a solution within timeout, return a greedy solution
    def greedy():
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
    
    return greedy()
