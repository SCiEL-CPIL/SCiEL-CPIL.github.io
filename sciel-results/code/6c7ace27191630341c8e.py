import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    num_diffs = row_len * (row_len - 1) // 2

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask, max_val=None):
        row = [0]
        x = 0
        limit = max_val if max_val is not None else 200
        while len(row) < row_len:
            x += 1
            if x > limit:
                return None
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (new_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_mask
        return row

    def find_valid_rows(used_mask, M, rng, node_budget=5000, deadline=None):
        candidates = []
        nodes = [0]
        
        def dfs(row, rmask, last):
            if deadline and time.time() > deadline:
                return
            if nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                if row[-1] <= M:
                    candidates.append((row[:], row_mask(row)))
                return
            
            hi = M
            lo = last + 1
            if lo > hi:
                return
            
            # Randomize the order of trying values to diversify search
            # But for finding *a* solution, ascending is often faster.
            # Let's use ascending.
            
            for x in range(lo, hi + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (x - a))
                if (new_mask & used_mask) or (new_mask & rmask):
                    continue
                row.append(x)
                dfs(row, rmask | new_mask, x)
                row.pop()
                if nodes[0] >= node_budget or (deadline and time.time() > deadline):
                    return

        dfs([0], 0, 0)
        return candidates

    def backtrack_solve(M, n_rows, used_mask, rng, deadline):
        rows = []
        
        def dfs(idx, used_mask):
            if time.time() > deadline:
                return False
            if idx == n_rows:
                return True
            
            candidates = find_valid_rows(used_mask, M, rng, node_budget=2000, deadline=deadline)
            
            if not candidates:
                return False
            
            # Heuristic: Sort by sum to prefer compact rows?
            candidates.sort(key=lambda x: sum(x[0]))
            
            for row, mask in candidates:
                if dfs(idx + 1, used_mask | mask):
                    rows.append(row)
                    return True
            
            return False

        if dfs(0, used_mask):
            return rows
        return None

    # Initial greedy to get a starting point / upper bound
    used_mask = 0
    initial_rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        if row is None:
            break
        used_mask |= row_mask(row)
        initial_rows.append(row)
    
    while len(initial_rows) < n_rows:
        row = greedy_row(used_mask, max_val=200)
        if row:
            used_mask |= row_mask(row)
            initial_rows.append(row)
        else:
            break

    current_best_rows = initial_rows
    current_best_scope = max(r[-1] for r in current_best_rows)
    
    # Binary search on Scope M
    lo = 100
    hi = current_best_scope
    
    if lo >= hi:
        lo = max(100, hi - 10)

    best_m = hi
    best_rows_for_m = current_best_rows
    
    # We will try to find a solution for M in [lo, hi]
    # Since feasibility is monotonic, if M works, M+1 works.
    # We want the smallest M.
    
    # To speed up, we can try a few specific M values first.
    # But binary search is standard.
    
    search_lo = lo
    search_hi = hi
    
    # Safety: ensure we don't loop forever
    max_iters = 20
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        if search_lo > search_hi:
            break
            
        mid = (search_lo + search_hi) // 2
        
        # Try to solve for M = mid
        rng = random.Random(mid)
        start_time = time.time()
        # Allocate time for this M check
        # If we have plenty of time, give it more.
        remaining = DEADLINE - time.time() - 5
        time_limit_for_m = min(15.0, remaining * 0.5)
        
        rows_m = backtrack_solve(mid, n_rows, 0, rng, start_time + time_limit_for_m)
        
        if rows_m is not None:
            best_m = mid
            best_rows_for_m = rows_m
            search_hi = mid - 1
        else:
            search_lo = mid + 1

    if best_rows_for_m and best_m < current_best_scope:
        return best_rows_for_m
    
    return current_best_rows
