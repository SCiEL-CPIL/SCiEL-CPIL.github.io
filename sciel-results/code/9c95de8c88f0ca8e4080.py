"""
Strategy: Minimum Remaining Values (MRV) Backtracking with Deterministic Restart.
Instead of random generation, we systematically explore the space of valid rows.
We prioritize rows that "cost" less in terms of blocking future options.
"""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Helper to compute differences for a row
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def is_valid(rows):
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    # Precompute all valid rows up to a certain scope limit
    # A row is valid internally if all its differences are distinct.
    # We store them as (max_val, mask, row_tuple)
    
    def generate_all_valid_rows(max_scope):
        valid_rows = []
        # Iterate over all combinations of size k from 1..max_scope
        # Prepend 0
        # Check internal validity
        for comb in combinations(range(1, max_scope + 1), k):
            row = (0,) + comb
            diffs = get_row_diffs(row)
            if len(diffs) == len(set(diffs)):
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                valid_rows.append((row[-1], mask, row))
        
        # Sort by max_val (scope), then by number of bits set (density)
        # Lower scope is better. 
        # For same scope, we might prefer rows that use "smaller" differences 
        # to leave larger differences open? Or rows that are "sparser"?
        # Let's just sort by scope.
        valid_rows.sort(key=lambda x: (x[0], x[1].bit_count()))
        return valid_rows

    # We will try increasing scope limits.
    # Counting bound for (7,5) is ~105. Best known 112.
    # Let's start at 105 and go up.
    
    # To make backtracking efficient, we need a good ordering of rows.
    # We can't precompute ALL rows up to 200 (too many).
    # But for scope ~110, C(110, 5) is ~ 50 million. Still too many to precompute fully in Python quickly.
    
    # Alternative: On-the-fly generation with caching?
    # Or, use a smarter heuristic: 
    # Instead of full backtracking on the entire space, we use a "Greedy with Conflict Resolution" 
    # but guided by a "Future Cost" heuristic.
    
    # Let's try a different approach: 
    # 1. Generate a large pool of candidate rows (e.g., 5000) with scope <= 120.
    # 2. Use a Maximal Independent Set style selection, but prioritize rows that 
    #    have the "lowest impact" on the remaining difference space.
    
    # Actually, the directive suggests "backtracking search that selects the next row element 
    # by minimizing the 'degree' of future constraint violations".
    
    # Let's implement a Backtracking Search that builds rows one by one.
    # State: list of selected rows, used_mask.
    # Heuristic: For the next row, among all valid rows (generated on the fly or from a pool) 
    # that don't conflict, pick the one that minimizes the "pressure" on the remaining rows.
    # Pressure metric: Number of differences used by this row that are "critical" for the remaining rows.
    # Since we don't know future rows, we approximate: pick the row that uses the "smallest" differences?
    # Or the row that leaves the most "large" differences free?
    
    # Let's try: 
    # 1. Generate a pool of candidate rows with scope <= 115.
    # 2. Sort them by scope.
    # 3. Backtracking:
    #    - If we have n_rows, return.
    #    - If time is up, return best found.
    #    - For each candidate row in the pool (sorted by scope, then by random shuffle for diversity in restarts):
    #        - If it conflicts, skip.
    #        - Recurse with this row added.
    
    # To make this fast, we need to prune. 
    # Pruning: If the number of remaining valid rows (from the pool that don't conflict with current mask) 
    # is less than the number of rows still needed, prune.
    
    def backtrack_search(pool, n_rows_needed, current_rows, used_mask, deadline, best_solution):
        if time.time() > deadline:
            return None
        
        if n_rows_needed == 0:
            return current_rows
        
        # Pruning heuristic: 
        # Count how many rows in the pool are compatible with used_mask.
        # If count < n_rows_needed, we are stuck.
        # This check is O(pool_size) per node, which might be slow if pool is large.
        # We can optimize by maintaining a list of compatible rows? 
        # Or just do a quick check.
        
        # Let's just iterate through the pool.
        # To ensure we don't miss solutions, we should consider all compatible rows.
        # But we want to find a solution with minimum scope.
        # The pool is sorted by scope. So the first solution we find in this DFS 
        # (if we explore in order) will be the one with the smallest possible max scope 
        # among the combinations of rows from this specific pool.
        
        for row in pool:
            # row is (max_val, mask, row_tuple)
            if used_mask & row[1]:
                continue
            
            # Check if we can still fill the remaining rows
            # Simple check: are there at least n_rows_needed rows in the pool 
            # (excluding current) that are compatible with (used_mask | row[1])?
            # This is expensive. Let's skip for now and rely on the pool being large enough.
            
            new_rows = current_rows + [row[2]]
            new_mask = used_mask | row[1]
            
            result = backtrack_search(pool, n_rows_needed - 1, new_rows, new_mask, deadline, best_solution)
            if result is not None:
                return result
            
            # If we found a solution, we return it. 
            # Since pool is sorted by scope, the first solution is optimal w.r.t the pool.
            # But we might want to try to improve it? 
            # No, the task is to minimize the largest entry. 
            # If we find a valid set with max scope S, any other set from this pool 
            # will have max scope >= S (because we picked the smallest scope rows first? 
            # Not necessarily, because we pick row 1, then row 2... 
            # The max scope of the set is max(max(r1), ..., max(rn)).
            # If we pick r1 with scope 100, r2 with scope 101, max is 101.
            # If we picked r1 with scope 101, r2 with scope 100, max is 101.
            # The order in the set doesn't matter for the max.
            # So finding ANY valid combination from the pool gives a scope.
            # To minimize the scope, we should restrict the pool to rows with scope <= S_limit.
            # We can binary search on S_limit.
            
        return None

    # Main Logic:
    # 1. Try increasing scope limits.
    # 2. For each limit, generate a pool of valid rows with max_val <= limit.
    # 3. Run backtracking to find a valid set of n_rows.
    # 4. If found, return it.
    # 5. If time runs out, return the best valid set found so far (or a greedy fallback).
    
    # To handle the "best known 112" and "bound 105", we start at 105.
    
    # However, generating the pool for limit=105 involves C(105, 5) ~ 30 million combinations.
    # This is too slow in Python to generate and filter every time.
    
    # Optimization: Precompute the pool for a slightly higher limit (e.g., 120) once, 
    # and then filter it for lower limits during the search?
    # C(120, 5) is ~ 190 million. Still too slow to generate in Python in 150s.
    
    # We need a faster way to generate candidate rows.
    # Random generation of valid rows is fast. 
    # Let's go back to random generation but with a smarter selection strategy.
    
    # New Strategy: 
    # 1. Generate a large pool of random valid rows (e.g., 20,000) with scope <= 115.
    # 2. Use a "Greedy with Backtracking" on this pool.
    #    - Sort pool by scope.
    #    - Try to pick rows one by one.
    #    - If stuck, backtrack.
    #    - Since the pool is random, we might miss the optimal combination.
    #    - To mitigate, we do multiple restarts with different random pools.
    
    # Let's implement this.
    
    def generate_random_pool(pool_size, max_scope, rng):
        pool = []
        seen = set()
        for _ in range(pool_size):
            # Sample k distinct values from 1..max_scope
            # To bias towards smaller scopes, we can sample from 1..max_scope 
            # but sort and take the first k? No, that's not random.
            # Just sample and sort.
            try:
                candidates = rng.sample(range(1, max_scope + 1), k)
            except ValueError:
                continue
            candidates.sort()
            row = (0,) + tuple(candidates)
            if row in seen:
                continue
            seen.add(row)
            
            diffs = get_row_diffs(row)
            if len(diffs) == len(set(diffs)):
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                pool.append((row[-1], mask, row))
        
        # Sort by scope
        pool.sort(key=lambda x: x[0])
        return pool

    best_solution = None
    best_scope = float('inf')
    
    # We will try a few scope limits.
    # Start with a tight limit, increase if not found.
    scope_limits = [105, 108, 110, 112, 115, 120]
    
    for limit in scope_limits:
        if time.time() > DEADLINE - 20:
            break
            
        # Try a few restarts for this limit
        for restart in range(5):
            if time.time() > DEADLINE - 10:
                break
                
            rng = random.Random(restart * 1000 + limit)
            pool = generate_random_pool(5000, limit, rng)
            
            if len(pool) < n_rows:
                continue
                
            # Backtracking search on this pool
            # We want to find a set of n_rows rows.
            # To make it faster, we can use a greedy first pass to get a candidate, 
            # then refine? 
            # Or just run the DFS.
            
            # DFS can be deep (n_rows=7). 
            # Branching factor is high (pool size). 
            # We need to prune aggressively.
            
            # Heuristic: 
            # When picking the next row, only consider rows that have the smallest scope 
            # and don't conflict. 
            # If we pick the smallest scope row, we are minimizing the current max.
            
            def dfs(idx, current_rows, used_mask):
                if time.time() > DEADLINE - 5:
                    return None
                
                if idx == n_rows:
                    return current_rows
                
                # Try rows in the pool starting from idx? 
                # No, we can pick any row from the pool for the next slot, 
                # as long as it doesn't conflict.
                # But to avoid permutations of the same set, we should enforce an order?
                # Yes, we should only pick rows with index > last_picked_index to avoid duplicates.
                # This assumes we pick rows in a fixed order from the pool.
                
                # However, the "best" set might not be contiguous in the sorted pool.
                # But if we sort by scope, picking non-contiguous rows is allowed.
                # To avoid permutations, we can say: the indices of the picked rows 
                # must be strictly increasing.
                
                for i in range(idx, len(pool)):
                    row = pool[i]
                    if used_mask & row[1]:
                        continue
                    
                    # Pruning: 
                    # If we pick this row, can we still pick n_rows - idx - 1 more rows 
                    # from pool[i+1:] that are compatible?
                    # This is hard to check fast.
                    
                    new_rows = current_rows + [row[2]]
                    new_mask = used_mask | row[1]
                    
                    res = dfs(i + 1, new_rows, new_mask)
                    if res is not None:
                        return res
                        
                return None
            
            result = dfs(0, [], 0)
            
            if result:
                scope = compute_scope(result)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = result
                    # If we found a solution with scope == limit, it's likely optimal for this limit.
                    # But we might find a smaller one in a tighter limit? 
                    # We are iterating limits in increasing order. 
                    # So the first limit that yields a solution is the minimal scope 
                    # achievable with this pool generation strategy.
                    # We can break out of the restart loop and the limit loop?
                    # Not necessarily, because a different random pool might allow a smaller scope 
                    # within the same limit? No, the limit is the max scope of the rows in the pool.
                    # If we found a solution with max scope S, and S <= limit, 
                    # it's a valid solution with scope S.
                    # Since we start with the smallest limit (105), if we find a solution, 
                    # its scope is <= 105. 
                    # If we don't find a solution for 105, we try 108.
                    # So the first solution found will have the smallest scope.
                    # We can return it immediately?
                    # Yes, because we are trying limits in increasing order.
                    # Any solution found at limit L has scope <= L.
                    # If we found one at 105, scope is <= 105.
                    # If we found one at 108, scope is <= 108.
                    # So the first success is the best.
                    
                    return result
        
    # If we didn't find a solution in the tight limits, we have a fallback.
    # The `best_solution` might be None if we never found one.
    # Let's do a final greedy attempt with a very large scope to ensure validity.
    
    if best_solution is None:
        # Fallback: Greedy construction with large scope
        rng = random.Random(42)
        rows = []
        used_mask = 0
        max_val = 200
        
        for _ in range(n_rows):
            found = False
            for _ in range(1000):
                candidates = rng.sample(range(1, max_val + 1), k)
                candidates.sort()
                row = (0,) + tuple(candidates)
                diffs = get_row_diffs(row)
                if len(diffs) != len(set(diffs)):
                    continue
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                
                if not (used_mask & mask):
                    rows.append(list(row))
                    used_mask |= mask
                    found = True
                    break
            
            if not found:
                # Increase max_val and retry
                max_val += 20
                
        if len(rows) == n_rows:
            return rows
        else:
            # Last resort: sequential construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
            
    return best_solution

# EVOLVE-BLOCK-END
