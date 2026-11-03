import heapq
import itertools
import time

def construct(n_rows, k):
    # Baseline: Greedy
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

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Precompute binomials for heuristic
    # C(n, 2) is number of diffs in a row of size n (which is k+1 elements)
    diffs_per_row = (k + 1) * k // 2
    
    def get_heuristic(i, used_mask):
        # Need to place diffs for remaining rows
        rows_rem = n_rows - i
        need = rows_rem * diffs_per_row
        
        if need == 0:
            return 0
            
        # Find the 'need'-th smallest positive integer not in used_mask
        # used_mask is a bitmask where bit d is 1 if d is used
        found = 0
        d = 1
        while found < need:
            if not (used_mask & (1 << d)):
                found += 1
                if found == need:
                    return d
            d += 1
        return d

    # Generate all valid rows up to best_scope
    # A row is a list of k+1 integers starting at 0, strictly increasing
    # Valid if all pairwise differences are distinct
    candidate_rows = []
    # We can generate rows recursively or iteratively
    # Given small k, we can use itertools.combinations for the non-zero elements
    # Elements are x1 < x2 < ... < xk where x0=0
    
    def generate_rows(limit):
        rows = []
        # Choose k elements from 1..limit
        for combo in itertools.combinations(range(1, limit + 1), k):
            row = [0] + list(combo)
            # Check distinct differences
            diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
            if valid:
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                rows.append((mask, max(row), row))
        return rows

    # To speed up, we might not want to generate ALL rows up to best_scope if it's large.
    # But for k=6, limit ~112, C(112, 6) is huge (~ 2e9). We cannot generate all.
    # We must generate rows on the fly or use a smarter search.
    
    # Alternative: A* where we build row by row, but prune aggressively.
    # The "Row as Unit" approach requires enumerating rows.
    # Let's stick to element-by-element A* but optimize the state and heuristic.
    
    # State: (f, h, i, row_idx, current_row, used_mask, path)
    # row_idx: how many rows completed
    # current_row: partial row for current row_idx
    
    start_time = time.time()
    timeout = 115.0
    
    # Heuristic: lower bound on max diff needed
    # We need `need` more diffs. The max diff must be at least the `need`-th smallest available diff.
    # This is admissible.
    
    def heuristic(i, row_len, used_mask):
        # i: number of completed rows
        # row_len: length of current partial row (starts with 0, so elements added = row_len - 1)
        # Diffs in current row so far: C(row_len, 2)
        # Diffs needed to complete current row: C(k+1, 2) - C(row_len, 2)
        # Diffs needed for remaining full rows: (n_rows - i - 1) * C(k+1, 2)
        
        c_k_plus_1_2 = (k + 1) * k // 2
        c_row_len_2 = row_len * (row_len - 1) // 2
        
        need_cur = c_k_plus_1_2 - c_row_len_2
        need_rem = (n_rows - i - 1) * c_k_plus_1_2
        total_need = need_cur + need_rem
        
        if total_need == 0:
            return 0
            
        # Find total_need-th smallest available diff
        found = 0
        d = 1
        # Optimization: if used_mask is 0, answer is total_need
        if used_mask == 0:
            return total_need
            
        while found < total_need:
            if not (used_mask & (1 << d)):
                found += 1
                if found == total_need:
                    return d
            d += 1
        return d

    # A* Search
    # State: (f, h, i, row, used_mask, path)
    # f = max(g, h)
    # g = max(max(r) for r in path) if path else 0, then max(g, max(row))
    
    def get_g(path, row):
        g = 0
        if path:
            g = max(max(r) for r in path)
        if row:
            g = max(g, max(row))
        return g

    start_h = heuristic(0, 1, 0)
    start_state = (start_h, start_h, 0, [0], 0, [])
    
    open_list = [start_state]
    visited = set()
    
    best_found = best_rows
    best_scope_val = best_scope
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, h, i, row, used_mask, path = heapq.heappop(open_list)
        
        state_key = (i, tuple(row), used_mask)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # If we found a complete solution
        if i == n_rows:
            scope = max(max(r) for r in path)
            if scope < best_scope_val:
                best_scope_val = scope
                best_found = path
                return path
        
        # Calculate current g
        g = 0
        if path:
            g = max(max(r) for r in path)
        if row:
            g = max(g, max(row))
            
        # If current g is already >= best_scope_val, no point continuing
        if g >= best_scope_val:
            continue
            
        # Expand state
        if len(row) < k + 1:
            last = row[-1]
            # Try next x
            # We can limit x to best_scope_val
            for x in range(last + 1, best_scope_val + 1):
                new_diffs = [x - a for a in row]
                valid = True
                new_mask = 0
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    new_row = row + [x]
                    new_used = used_mask | new_mask
                    new_i = i
                    new_path = path
                    
                    new_g = max(g, x)
                    new_h = heuristic(new_i, len(new_row), new_used)
                    new_f = max(new_g, new_h)
                    
                    if new_f < best_scope_val:
                        child_state = (new_f, new_h, new_i, new_row, new_used, new_path)
                        heapq.heappush(open_list, child_state)
        else:
            # Row is full, move to next row
            if i < n_rows - 1:
                new_i = i + 1
                new_row = [0]
                new_used = used_mask
                new_path = path + [row]
                
                new_g = g # max of path is same as g since row is added to path
                # Actually g was max(max(path), max(row)). Now path includes row.
                # So new_g is just max(max(new_path)) which is same as g.
                
                new_h = heuristic(new_i, 1, new_used)
                new_f = max(new_g, new_h)
                
                if new_f < best_scope_val:
                    child_state = (new_f, new_h, new_i, new_row, new_used, new_path)
                    heapq.heappush(open_list, child_state)

    return best_found
