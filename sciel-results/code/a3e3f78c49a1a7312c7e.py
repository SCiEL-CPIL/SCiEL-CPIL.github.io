import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 140.0  # Safety margin under 150s kill
    best_max = float('inf')
    best_rows = None

    # --- Helper Functions ---
    def get_diffs(row):
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def greedy_row(used, k_val, rng, max_val=None):
        """Build one row greedily, choosing the smallest available next mark."""
        row = [0]
        x = 0
        while len(row) < k_val + 1:
            x += 1
            if max_val is not None and x > max_val:
                break
            # Occasionally jump ahead to create diversity
            if rng.random() < 0.15:
                jump = rng.randint(1, 4)
                x += jump
                if max_val is not None and x > max_val:
                    break
            
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row

    # --- SA Optimization ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len
        
        # Precompute diffs per row
        row_diffs = [get_diffs(r) for r in rows]
        
        # Global diff counts
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            sum_entries = sum(sum(r) for r in rows)
            max_val = max(r[-1] for r in rows)
            # Weight collisions heavily, then sum entries (smooth surrogate for scope reduction), then max
            return collisions * 10000 + sum_entries + max_val

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 80.0
        cooling = 0.9998
        start_time = time.time()
        rng = random

        while time.time() - start_time < time_limit and time.time() < deadline - 5.0:
            move_type = rng.random()
            
            if move_type < 0.7:
                # Move: Rebuild a random row
                r_idx = rng.randint(0, n - 1)
                old_row = rows[r_idx]
                
                # Remove old diffs
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                # Rebuild row with some randomness
                used = set(diff_counts.keys())
                # Allow slight expansion to escape local minima
                max_limit = bound + rng.randint(0, 5)
                new_row = greedy_row(used, k, rng, max_limit)
                
                if len(new_row) < L:
                    # Restore old row
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    continue
                
                # Add new diffs
                new_diffs = get_diffs(new_row)
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_diffs
                
            else:
                # Move: Shift an entry in a random row
                r_idx = rng.randint(0, n - 1)
                i_idx = rng.randint(1, L - 1)
                
                row = rows[r_idx]
                old_val = row[i_idx]
                
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 10
                upper = min(upper, bound + 20)
                
                if lower > upper:
                    continue
                    
                new_val = rng.randint(lower, upper)
                if new_val == old_val:
                    continue
                
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                row[i_idx] = new_val
                new_diffs = get_diffs(row)
                row_diffs[r_idx] = new_diffs
                
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if rng.random() < 0.5:
                        current_cost = new_cost
                    else:
                        # Revert logic would be complex here due to row rebuild vs entry shift
                        # For simplicity in this combined structure, we just accept equal cost moves 
                        # or rely on the fact that we track best separately. 
                        # However, to maintain SA properties, we should revert if we want to be strict.
                        # Given the complexity of reverting a full row rebuild vs entry shift, 
                        # and that we track 'best_local_rows' separately, we can just let current state drift 
                        # if it's equal, or revert. Let's revert to be safe.
                        if move_type < 0.7:
                            # Revert row rebuild
                            rows[r_idx] = old_row
                            row_diffs[r_idx] = old_diffs
                            for d in new_diffs:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in old_diffs:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        else:
                            # Revert entry shift
                            row[i_idx] = old_val
                            for d in new_diffs:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in old_diffs:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                            row_diffs[r_idx] = old_diffs
                else:
                    if delta < 5000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if rng.random() < p:
                        current_cost = new_cost
                    else:
                        # Revert
                        if move_type < 0.7:
                            rows[r_idx] = old_row
                            row_diffs[r_idx] = old_diffs
                            for d in new_diffs:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in old_diffs:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        else:
                            row[i_idx] = old_val
                            for d in new_diffs:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in old_diffs:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                            row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.1:
                T = 0.1

        # High-precision polish pass
        T = 5.0
        cooling = 0.999
        polish_start = time.time()
        polish_limit = min(5.0, deadline - time.time() - 2.0)
        
        # Reset to best found so far
        rows = [r[:] for r in best_local_rows]
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        while time.time() - polish_start < polish_limit and time.time() < deadline - 2.0:
            r_idx = rng.randint(0, n - 1)
            i_idx = rng.randint(1, L - 1)
            
            row = rows[r_idx]
            old_val = row[i_idx]
            
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else best_local_max + 5
            upper = min(upper, best_local_max + 5)
            
            if lower > upper:
                continue
                
            new_val = rng.randint(lower, upper)
            if new_val == old_val:
                continue
            
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta < 200:
                    p = math.exp(-delta / T)
                else:
                    p = 0.0
                
                if rng.random() < p:
                    current_cost = new_cost
                else:
                    row[i_idx] = old_val
                    for d in new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # --- CSP Backtracking ---
    def csp_construct(target_M, time_limit):
        used = set()
        rows = []
        start_time = time.time()
        
        def dfs(row_idx):
            if time.time() - start_time > time_limit or time.time() > deadline - 10.0:
                return False
            
            if row_idx == n_rows:
                return True
            
            attempts = 0
            while attempts < 50:
                attempts += 1
                if target_M < k:
                    return False
                
                pts = random.sample(range(1, target_M + 1), k)
                pts.sort()
                row = [0] + pts
                
                diffs = get_diffs(row)
                if len(set(diffs)) != len(diffs):
                    continue
                
                if any(d in used for d in diffs):
                    continue
                
                rows.append(row)
                used.update(diffs)
                
                if dfs(row_idx + 1):
                    return True
                else:
                    rows.pop()
                    for d in diffs:
                        used.discard(d)
            
            return False
        
        if dfs(0):
            return rows
        return None

    # --- Main Loop ---
    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)

    while time.time() < deadline - 5.0:
        # 1. Try CSP to find a valid solution with smaller scope
        # Start from current best_max and go down
        for target in range(best_max, best_max - 20, -1):
            if time.time() > deadline - 10.0:
                break
            csp_time = min(2.0, deadline - time.time() - 10.0)
            if csp_time <= 0:
                break
            
            result = csp_construct(target, csp_time)
            if result is not None:
                new_max = max(r[-1] for r in result)
                if new_max < best_max:
                    best_rows = result
                    best_max = new_max
                    break
        
        # 2. Polish with SA
        sa_time = min(10.0, deadline - time.time() - 2.0)
        if sa_time <= 0:
            break
        
        # Initialize near best solution
        init_rows = [r[:] for r in best_rows]
        # Add slight randomization
        for _ in range(3):
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            row = init_rows[r_idx]
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 5
            if lower <= upper:
                init_rows[r_idx][i_idx] = random.randint(lower, upper)
        
        local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
        
        if local_cost < 10000: # Valid
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

    return best_rows
