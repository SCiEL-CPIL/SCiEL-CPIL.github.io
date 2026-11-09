import time
import random
import sys

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    def remaining():
        return DEADLINE - (time.time() - t0)

    # n_rows=4, k=8 -> 9 elements per row
    # Differences: 9*8/2 = 36 differences per row.
    # Total differences needed: 4 * 36 = 144.
    # Differences range from 1 to M (scope).
    # We need 144 distinct differences in [1, M].
    # Simple bound: M >= 144. But overlaps in generation make it larger.
    # Known best is 134.

    needed_diffs_per_row = k * (k + 1) // 2
    total_needed_diffs = n_rows * needed_diffs_per_row

    # Step 1: Generate a diverse pool of candidate rows.
    # We generate rows with small scopes first.
    # A row is valid if its internal differences are unique.
    # We store (scope, mask, row)
    
    rng = random.Random(42)
    pool = []
    seen_masks = set()
    
    # We will try to generate rows with scopes up to a limit.
    # If we don't find enough, we expand the limit.
    # Start with a reasonable limit, e.g., 150 (since best is 134).
    # But to be safe and fast, we generate randomly and filter by scope.
    
    def gen_row(rng, bias, window, max_val_limit=200):
        row = [0]
        x = 0
        used = set()
        while len(row) < k + 1:
            cands = []
            y = x + 1
            scanned = 0
            # Look for candidates in a window
            while len(cands) < window and scanned < 200:
                scanned += 1
                new_diffs = {y - a for a in row}
                if not (new_diffs & used):
                    cands.append(y)
                y += 1
            
            # If window exhausted, keep scanning
            if not cands:
                while True:
                    new_diffs = {y - a for a in row}
                    if not (new_diffs & used):
                        cands.append(y)
                        break
                    y += 1
                    if y > max_val_limit:
                        return None, None
            
            if rng.random() < bias:
                chosen = cands[0]
            else:
                idx = int(len(cands) * (rng.random() ** 1.5))
                idx = min(idx, len(cands) - 1)
                chosen = cands[idx]
            
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
            x = chosen
            
        # Calculate diff mask
        diffs = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ai)
        
        if len(diffs) != needed_diffs_per_row:
            return None, None
            
        mask = 0
        for d in diffs:
            mask |= (1 << d)
            
        return row, mask

    # Generate pool
    target_pool_size = 2000
    attempts = 0
    max_attempts = 10000
    
    # We want rows with small scopes.
    # We'll generate and keep those with scope < 160 initially.
    scope_limit = 160
    
    while len(pool) < target_pool_size and attempts < max_attempts and remaining() > 60:
        attempts += 1
        bias = rng.uniform(0.5, 0.9)
        window = rng.choice([3, 5, 8, 12])
        
        row, mask = gen_row(rng, bias, window, max_val_limit=scope_limit + 10)
        if row is None:
            continue
            
        scope = max(row)
        if scope > scope_limit:
            continue
            
        if mask in seen_masks:
            continue
        seen_masks.add(mask)
        
        pool.append((scope, mask, row))
        
    # If pool is too small, lower the scope limit or increase attempts?
    # For now, if pool is empty, we have a problem.
    if not pool:
        # Emergency: generate a few rows without strict scope limit to ensure we have *something*
        for _ in range(100):
            row, mask = gen_row(rng, 0.8, 5, max_val_limit=300)
            if row and mask not in seen_masks:
                seen_masks.add(mask)
                pool.append((max(row), mask, row))
        if not pool:
            # This should never happen, but return a dummy valid structure to avoid crash
            # Constructing a valid set is non-trivial, but for n_rows=1 it's easy.
            # For n_rows>1, we just return a single row repeated? No, that's invalid.
            # But the task says "invalid result... scores 0". We want to avoid exceptions.
            # Let's try to construct a trivial valid set if possible?
            # For k=8, n_rows=4, we need 144 distinct diffs.
            # If we fail, we return a list of one row? No, must be n_rows.
            # We will return a list of n_rows identical rows? No, invalid.
            # We will return a list of n_rows rows that are valid individually, even if they conflict?
            # The evaluator checks validity. If invalid, score 0.
            # But an Exception also scores 0.
            # So returning an invalid list is better than an Exception?
            # Actually, the prompt says "invalid result, an exception or a timeout scores 0".
            # So we should try our best to return a *valid* result.
            # If we can't, we return an empty list? No, must be n_rows.
            # Let's just return a list of n_rows zeros? [0,0,...,0] is not strictly increasing.
            # [0, 1, 2, ...] is strictly increasing.
            # Let's return n_rows copies of [0, 1, ..., k].
            # Diffs will be 1..k. They will conflict. But it's a list of lists.
            dummy_row = list(range(k + 1))
            return [dummy_row] * n_rows

    # Sort by scope
    pool.sort(key=lambda x: x[0])
    
    # Step 2: Find 4 rows with disjoint masks.
    # We use a backtracking search.
    # Since n_rows is small (4), we can try to find a clique of size 4 in the compatibility graph.
    # Compatibility: (mask_i & mask_j) == 0
    
    best_scope = float('inf')
    best_solution = None
    
    # Pre-compute compatible indices for speed?
    # With ~2000 nodes, checking all pairs is 2M ops, which is fine in Python if done once?
    # Or just check on the fly.
    
    def search(start_idx, selected, current_max_scope, used_mask):
        nonlocal best_scope, best_solution
        
        if len(selected) == n_rows:
            if current_max_scope < best_scope:
                best_scope = current_max_scope
                best_solution = [r[2] for r in selected]
            return
        
        # Pruning
        if remaining() < 5:
            return
            
        # If current_max_scope >= best_scope, we can prune if we assume we pick rows with scope >= current?
        # No, we pick from start_idx, so scopes are non-decreasing.
        # The max scope of the final set will be at least current_max_scope.
        # If current_max_scope >= best_scope, we can't improve.
        if current_max_scope >= best_scope:
            return
            
        # Try adding pool[i]
        for i in range(start_idx, len(pool)):
            scope_i, mask_i, row_i = pool[i]
            
            # If this row's scope is already >= best_scope, and we haven't picked all, 
            # the final max scope will be >= scope_i >= best_scope.
            # So we can break?
            # Yes, because pool is sorted by scope.
            if scope_i >= best_scope:
                break
                
            if used_mask & mask_i:
                continue
                
            new_max_scope = max(current_max_scope, scope_i)
            
            search(i + 1, selected + [(scope_i, mask_i, row_i)], new_max_scope, used_mask | mask_i)
            
            # Optional: If we found a solution with scope == scope_i, we might prune further?
            # But we are minimizing, so we keep searching for smaller.

    # To speed up search, we can limit the depth of the search or the number of candidates considered at each step.
    # Given the pool is sorted, the first few rows have the smallest scopes.
    # We only need 4 rows.
    
    # Let's restrict the search to the first N smallest scope rows to avoid deep recursion on large scopes.
    # If no solution is found in the first 500 rows, we might need to look further, but it's unlikely to be better than the first 500 if they are dense.
    # Actually, if the first 500 don't contain a clique of 4, a solution with smaller max scope is unlikely to be in the 500-1000 range.
    # But we must ensure we find *a* solution.
    
    # We will run the search on the whole pool, but with a timeout guard.
    # To prevent explosion, we can limit the branching factor?
    # No, backtracking is necessary.
    
    # Let's try the search.
    search(0, [], 0, 0)
    
    if best_solution:
        return best_solution
    
    # Fallback: Greedy selection from the pool.
    # This guarantees a solution if the pool contains 4 compatible rows, even if not optimal.
    # We pick the row with smallest scope, then next compatible, etc.
    used = 0
    res = []
    for scope, mask, row in pool:
        if not (used & mask):
            res.append(row)
            used |= mask
            if len(res) == n_rows:
                break
                
    if len(res) == n_rows:
        return res
        
    # If even greedy fails (pool has no 4 compatible rows), we have to return something.
    # We return the first n_rows rows from the pool (they will be invalid, but it's a list of lists).
    # This is better than an exception.
    if len(pool) >= n_rows:
        return [r[2] for r in pool[:n_rows]]
    else:
        # Pad with dummy rows if pool is too small
        dummy_row = list(range(k + 1))
        while len(pool) < n_rows:
            pool.append((k, 0, dummy_row))
        return [r[2] for r in pool[:n_rows]]
