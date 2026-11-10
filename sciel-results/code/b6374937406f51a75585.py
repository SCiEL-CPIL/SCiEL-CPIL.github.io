import time
import random
import itertools

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Greedy/Randomized Search for Initial Solution
    best_rows = None
    best_scope = float('inf')
    
    # Run several randomized greedy restarts
    # Time budget for this phase: 20s
    greedy_deadline = start_time + 20
    for _ in range(1000): # Max iterations
        if time.time() > greedy_deadline:
            break
        rows = try_random_greedy(n_rows, k)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                
    if best_rows is None:
        # Fallback to deterministic greedy if random failed (unlikely)
        best_rows = try_deterministic_greedy(n_rows, k)
        best_scope = max(max(r) for r in best_rows)

    # 2. Backtracking Refiner
    # Try to lower the scope
    # We can only pre-generate if M is small enough
    # Threshold: 5 million rows
    # k=5, M=115 -> 2.5M. M=120 -> 4.8M.
    
    # We will try M from best_scope down to theoretical_min
    theoretical_min = n_rows * (k * (k + 1) // 2)
    
    # If best_scope is too high for pre-generation, we might skip refiner or use a different method
    # But for 7,5, best_scope should be ~112.
    
    current_m = best_scope
    while current_m >= theoretical_min:
        if time.time() > deadline - 10: # 10s buffer
            break
            
        # Check if pre-generation is feasible
        # Calculate C(M, k)
        # If too big, break (assume no solution for smaller M or too slow to check)
        # Actually, if M is big, we just accept the greedy result.
        if is_pre_generation_feasible(current_m, k):
            solution = try_backtracking(n_rows, k, current_m, time.time() + 10)
            if solution is not None:
                best_rows = solution
                best_scope = current_m
            else:
                # If no solution found within time, break?
                # It's possible a solution exists but search was too slow.
                # However, for this problem size, if 112 fails, 111 likely fails or is harder.
                # We break to save time for lower M? No, lower M is harder to find in pre-gen?
                # Actually, smaller M means fewer rows, so pre-gen is faster, but solution space is tighter.
                # If 112 has no solution, 111 might.
                # So we continue to current_m - 1.
                pass
        else:
            # M is too large for pre-gen.
            # We cannot easily verify if a smaller M is possible without pre-gen.
            # So we stop.
            break
            
        current_m -= 1

    return best_rows

def try_random_greedy(n_rows, k):
    # Returns list of rows or None
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        # Randomized construction
        # To make it fast, we can shuffle the candidate order
        # But we need to find a valid row.
        # Simple approach: try to build row, if stuck, fail.
        # To increase success, we can randomize the choice of next mark.
        
        # However, pure random might fail often.
        # Let's use a greedy with random tie-breaking.
        # At each step, collect valid next marks, pick one randomly.
        
        current_row = [0]
        while len(current_row) < k + 1:
            last = current_row[-1]
            valid_nexts = []
            # Max possible value? We don't have a fixed M here.
            # We can just go up to some limit, e.g., 200.
            # Or, we can determine the limit dynamically.
            # Let's cap at 200 for greedy.
            for x in range(last + 1, 201):
                new_diffs = {x - a for a in current_row}
                if not new_diffs & used:
                    valid_nexts.append(x)
            
            if not valid_nexts:
                return None # Failed
            
            # Pick randomly
            next_val = random.choice(valid_nexts)
            current_row.append(next_val)
            used.update({next_val - a for a in current_row[:-1]})
        
        rows.append(current_row)
    return rows

def try_deterministic_greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    return rows

def is_pre_generation_feasible(M, k):
    # Calculate C(M, k)
    # Approximation or exact?
    # For k=5, M=120, C(120,5) ~ 4.8M
    # Limit to 5M
    # Use math.comb
    try:
        import math
        return math.comb(M, k) < 5_000_000
    except ImportError:
        # Fallback
        if k == 5:
            return M < 125
        return False

def try_backtracking(n_rows, k, M, deadline):
    # Generate all valid rows
    # Row is (mask, row_tuple)
    rows = []
    # itertools.combinations(range(1, M+1), k)
    # Prepend 0
    
    # Optimization: 
    # We can prune rows during generation?
    # No, we need all.
    
    # Generation
    for combo in itertools.combinations(range(1, M + 1), k):
        row = (0,) + combo
        # Check if strictly increasing (it is)
        # Compute mask
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        rows.append((mask, row))
        
    if time.time() > deadline:
        return None

    # DFS
    # To speed up, sort rows?
    # Maybe sort by mask bit count? (All have same bit count)
    # Maybe sort by mask value?
    
    # We need to find n_rows disjoint rows.
    
    # State: idx, used_mask
    # We can use a recursive function
    
    def dfs(idx, used_mask, current_rows):
        if time.time() > deadline:
            return None
        
        if idx == n_rows:
            return current_rows
        
        # Try each row
        # Optimization: If we have many rows, iterating all is slow.
        # We can filter: if mask & used_mask != 0, skip.
        
        # To improve, we can remove used rows from the list?
        # No, rows are not unique by mask? 
        # Two different rows can have the same mask?
        # Yes, e.g., [0,1,3] diffs {1,2,3} and [0,2,3] diffs {1,2,3}?
        # [0,1,3] diffs: 1, 3, 2. Set {1,2,3}.
        # [0,2,3] diffs: 2, 3, 1. Set {1,2,3}.
        # So yes, same mask possible.
        # But if we pick one, we can't pick the other (same mask).
        # So we can group by mask?
        # If we group by mask, we only need to try one row per mask.
        # This reduces the search space significantly!
        
        # Let's restructure:
        # 1. Generate all rows.
        # 2. Group by mask.
        # 3. For each mask, keep one representative row.
        # 4. DFS on the unique masks.
        
        # This is a huge optimization.
        
        # We will implement this grouping inside try_backtracking before DFS.
        
        pass # Placeholder

    # Implement Grouping
    mask_to_row = {}
    for mask, row in rows:
        if mask not in mask_to_row:
            mask_to_row[mask] = row
            
    unique_masks = list(mask_to_row.keys())
    
    # DFS on unique_masks
    def dfs(idx, used_mask, current_rows):
        if time.time() > deadline:
            return None
        
        if idx == n_rows:
            return current_rows
        
        for mask in unique_masks:
            if mask & used_mask:
                continue
                
            new_mask = used_mask | mask
            # Recurse
            res = dfs(idx + 1, new_mask, current_rows + [mask_to_row[mask]])
            if res is not None:
                return res
                
        return None

    return dfs(0, 0, [])
