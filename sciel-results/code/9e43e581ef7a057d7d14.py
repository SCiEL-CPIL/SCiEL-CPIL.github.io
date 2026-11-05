import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    
    Strategy:
    1. Find a baseline solution using randomized greedy.
    2. Generate a pool of candidate rows.
    3. Use Simulated Annealing / Local Search on the pool to minimize the max entry.
    """
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to generate a row with greedy/randomized approach
    def gen_row(used_diffs, max_val, randomize=True):
        row = [0]
        for _ in range(k):
            last_val = row[-1]
            candidates = []
            # Search for valid next value
            # To speed up, we can limit the search range or use step
            # We want to keep values small
            for x in range(last_val + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    candidates.append(x)
                    if len(candidates) > 20 and randomize:
                        break
            
            if not candidates:
                return None
            
            if randomize:
                # Bias towards smaller x
                # Weighted random choice? Or just pick from top few
                candidates.sort()
                if random.random() < 0.5:
                    x = candidates[0]
                else:
                    x = random.choice(candidates[:min(5, len(candidates))])
            else:
                x = min(candidates)
            
            row.append(x)
            used_diffs |= {x - a for a in row[:-1]} # Wait, need to update used_diffs with new diffs
            # Actually, used_diffs is passed by reference? No, it's a set.
            # We should update the local set or return it.
            # Let's return the mask or set.
            
        return row

    # Better gen_row that returns mask and max_val
    def gen_row_mask(used_mask, max_val, randomize=True):
        row = [0]
        mask = 0
        for _ in range(k):
            last_val = row[-1]
            candidates = []
            for x in range(last_val + 1, max_val + 1):
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if valid:
                    candidates.append((x, new_diffs_mask))
                    if len(candidates) > 10 and randomize:
                        break
            
            if not candidates:
                return None, 0, 0
            
            if randomize:
                candidates.sort(key=lambda item: item[0])
                if random.random() < 0.6:
                    x, new_mask = candidates[0]
                else:
                    x, new_mask = random.choice(candidates[:min(3, len(candidates))])
            else:
                x, new_mask = min(candidates, key=lambda item: item[0])
            
            row.append(x)
            mask |= new_mask
            
        return row, max(row), mask

    # 1. Baseline
    best_scope = float('inf')
    best_result = None
    
    # Quick greedy to get a bound
    for _ in range(100):
        if time.time() - start_time > time_limit * 0.1:
            break
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            res = gen_row_mask(used_mask, 200, randomize=True)
            if res[0] is None:
                success = False
                break
            row, max_v, mask = res
            rows.append(row)
            used_mask |= mask
            if max_v > best_scope:
                # If we exceed current best, we can stop early?
                # But we need to finish the row to check validity?
                # Actually, if max_v > best_scope, this restart might not improve.
                # But it might fail later.
                pass
        
        if success:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_result = [r[:] for r in rows]
        
        if best_scope <= 105:
            break

    if best_result is None:
        # Fallback
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            res = gen_row_mask(used_mask, 500, randomize=False)
            row, max_v, mask = res
            rows.append(row)
            used_mask |= mask
        return rows

    # 2. Pool Generation and Local Search
    # We will try to improve best_scope using SA on a pool.
    # Pool size: 2000
    pool_size = 2000
    
    # We generate rows with max_val <= best_scope (or slightly less)
    # If we find a row with max_val < best_scope, it's good.
    
    # We will run SA for the remaining time.
    
    # Initial solution for SA: Use the best_result rows' masks?
    # No, we need to map them to the pool.
    # It's easier to just start SA with a random valid set from the pool, 
    # or use the best_result as the initial state if we can represent it.
    
    # Let's generate the pool first.
    pool = []
    # To ensure diversity, we can generate rows with different max_val bounds.
    # But for now, just randomize.
    
    for _ in range(pool_size):
        if time.time() - start_time > time_limit * 0.3:
            break
        # Try to generate a row that is "tight" (small max val)
        # We don't have a global used_mask here, so we generate standalone rows.
        # Wait, the pool rows must be valid *individually*. 
        # The disjointness is checked during SA.
        # So we just generate random valid rows (starting with 0).
        
        # To make them "good", we can use a greedy approach with a small max_val limit?
        # If we limit max_val, we might not find rows.
        # Let's just generate random rows up to best_scope.
        
        res = gen_row_mask(0, best_scope, randomize=True)
        if res[0] is not None:
            row, max_v, mask = res
            pool.append((row, max_v, mask))
    
    if not pool:
        return best_result

    # Initial state for SA
    # Pick n_rows random rows from pool that are disjoint?
    # This is hard. 
    # Instead, let's use the best_result as the initial state?
    # But best_result rows might not be in the pool.
    # We can add best_result rows to the pool.
    for r in best_result:
        mask = 0
        for i in range(len(r)):
            for j in range(i):
                d = r[i] - r[j]
                mask |= (1 << d)
        pool.append((r, r[-1], mask))
    
    # Now, find an initial valid set from the pool.
    # Greedy:
    current_sol = [] # list of indices in pool
    used_mask = 0
    for i in range(len(pool)):
        if len(current_sol) >= n_rows:
            break
        if (pool[i][2] & used_mask) == 0:
            current_sol.append(i)
            used_mask |= pool[i][2]
    
    if len(current_sol) < n_rows:
        # Failed to find initial set in pool. 
        # This can happen if pool is small or bad.
        # Fallback to best_result.
        return best_result

    current_max = max(pool[i][1] for i in current_sol)
    best_sa_max = current_max
    best_sa_sol = current_sol[:]
    
    # SA Parameters
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    # Precompute masks for faster access?
    pool_masks = [p[2] for p in pool]
    pool_maxs = [p[1] for p in pool]
    
    while time.time() - start_time < time_limit * 0.9 and T > min_T:
        # Pick a row to replace
        idx_in_sol = random.randint(0, len(current_sol) - 1)
        old_idx = current_sol[idx_in_sol]
        old_mask = pool_masks[old_idx]
        
        # New mask without old row
        temp_used = used_mask ^ old_mask
        
        # Pick a new row from pool
        new_idx = random.randint(0, len(pool) - 1)
        
        # Check if new_idx is already in current_sol
        if new_idx in current_sol:
            continue
            
        new_mask = pool_masks[new_idx]
        
        if (temp_used & new_mask) == 0:
            # Valid move
            new_used = temp_used | new_mask
            new_max = max(pool_maxs[new_idx], max(pool_maxs[i] for i in current_sol if i != old_idx))
            
            # Acceptance criterion
            if new_max < current_max:
                # Always accept improvement
                current_sol[idx_in_sol] = new_idx
                used_mask = new_used
                current_max = new_max
                
                if current_max < best_sa_max:
                    best_sa_max = current_max
                    best_sa_sol = current_sol[:]
            else:
                # Deterioration or same
                # Accept with probability exp(-(new_max - current_max) / T)
                # Since we want to minimize, delta = new_max - current_max >= 0
                if random.random() < pow(2.71828, -(new_max - current_max) / T):
                    current_sol[idx_in_sol] = new_idx
                    used_mask = new_used
                    current_max = new_max
        
        T *= cooling
    
    # Reconstruct best solution
    if best_sa_max < best_scope:
        best_scope = best_sa_max
        best_result = [pool[i][0][:] for i in best_sa_sol]
        
    return best_result
