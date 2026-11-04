import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def get_mask(diffs):
        mask = 0
        for d in diffs:
            mask |= (1 << d)
        return mask

    def is_valid_row_internal(row):
        diffs = get_diffs(row)
        return len(set(diffs)) == len(diffs)

    def generate_random_row(M):
        # Pick k distinct values from 1..M
        if M < k:
            return None
        try:
            vals = random.sample(range(1, M + 1), k)
            vals.sort()
            row = [0] + vals
            if is_valid_row_internal(row):
                return row
        except:
            pass
        return None

    def solve_for_scope(M, time_budget):
        # Generate a pool of candidate rows
        pool = []
        pool_masks = []
        attempts = 0
        # Target pool size: 500 is a good balance
        target_pool_size = 500
        
        while len(pool) < target_pool_size and attempts < 50000 and time.time() - start_time < time_limit - 5:
            row = generate_random_row(M)
            if row:
                diffs = get_diffs(row)
                mask = get_mask(diffs)
                pool.append(row)
                pool_masks.append(mask)
            attempts += 1
            
        if len(pool) < n_rows:
            return None
            
        # Calculate conflict counts to order candidates
        # conflict_count[i] is number of other rows that conflict with row i
        n_pool = len(pool)
        conflict_counts = [0] * n_pool
        for i in range(n_pool):
            mi = pool_masks[i]
            for j in range(i + 1, n_pool):
                if mi & pool_masks[j]:
                    conflict_counts[i] += 1
                    conflict_counts[j] += 1
        
        # Sort indices by conflict count (ascending: pick less conflicting rows first)
        indices = list(range(n_pool))
        indices.sort(key=lambda x: conflict_counts[x])
        
        # Create sorted pool and masks
        sorted_pool = [pool[i] for i in indices]
        sorted_masks = [pool_masks[i] for i in indices]
        
        def backtrack(idx, used_mask, current_rows):
            if time.time() - start_time > time_limit - 2:
                return None
            
            if idx == n_rows:
                return list(current_rows)
                
            for i in range(len(sorted_pool)):
                if sorted_masks[i] & used_mask:
                    continue
                    
                new_mask = used_mask | sorted_masks[i]
                res = backtrack(idx + 1, new_mask, current_rows + [sorted_pool[i]])
                if res:
                    return res
            return None
        
        # Try multiple restarts with different random shuffles of the *tail* of the sorted list?
        # Or just shuffle the whole thing?
        # The sorting helps, but randomization helps escape local optima in the search tree.
        # We will shuffle the sorted list slightly or just rely on the sorted order for the first try,
        # then randomize for subsequent tries.
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() - start_time > time_limit - 2:
                break
            
            if r == 0:
                # Use sorted order
                sp = sorted_pool
                sm = sorted_masks
            else:
                # Randomize
                sp = sorted_pool[:]
                sm = sorted_masks[:]
                random.shuffle(sp)
                random.shuffle(sm)
                
            # Re-bind variables for closure
            def backtrack_r(idx, used_mask, current_rows, sp, sm):
                if time.time() - start_time > time_limit - 2:
                    return None
                if idx == n_rows:
                    return list(current_rows)
                for i in range(len(sp)):
                    if sm[i] & used_mask:
                        continue
                    new_mask = used_mask | sm[i]
                    res = backtrack_r(idx + 1, new_mask, current_rows + [sp[i]], sp, sm)
                    if res:
                        return res
                return None
            
            res = backtrack_r(0, 0, [], sp, sm)
            if res:
                return res
                
        return None

    # Main loop
    # Start from lower bound
    lower_bound = n_rows * (k * (k + 1) // 2)
    
    # We will try scopes from lower_bound upwards
    for M in range(lower_bound, 151):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try to find a solution for this M with a time budget
        # Allocate 5s per M
        sol = solve_for_scope(M, 5.0)
        if sol:
            best_rows = sol
            best_scope = M
            # Since we iterate from low to high, the first valid M is the minimum
            break

    if best_rows is None:
        # Fallback: Greedy construction
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
