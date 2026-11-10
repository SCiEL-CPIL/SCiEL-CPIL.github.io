import time
import random
import math

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def global_max(rows_list):
        m = 0
        for r in rows_list:
            if r[-1] > m:
                m = r[-1]
        return m

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            row = [0] + list(range(1, k + 1))  # fallback
        rows.append(row)
        used |= row_diffs(row)
    
    current_max = global_max(rows)
    best_rows = rows[:]
    best_max = current_max

    # Phase 2: Precompute pool of all valid rows up to baseline_max
    # Each entry: (max_val, frozenset_of_diffs, tuple_row)
    pool = []
    pool_start = time.time()
    pool_timeout = 25.0  # seconds for pool generation
    
    def generate_pool(limit, k, time_limit):
        """Generate all valid rows starting at 0 with k+1 elements, max <= limit."""
        start_t = time.time()
        results = []
        bit_of = [1 << d for d in range(limit + 1)]
        
        def backtrack(current_row, current_mask, last_val):
            if time.time() - start_t > time_limit:
                return False
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                # Compute the diff set for this row
                diffs = set()
                for i in range(len(current_row)):
                    a = current_row[i]
                    for j in range(i + 1, len(current_row)):
                        diffs.add(current_row[j] - a)
                results.append((max_val, frozenset(diffs), tuple(current_row)))
                return True
            
            # Pruning: we need (k+1 - len(current_row)) more elements
            # The minimum possible max is last_val + (needed)
            needed = (k + 1) - len(current_row)
            min_possible_max = last_val + needed
            if min_possible_max > limit:
                return True
            
            for x in range(last_val + 1, limit + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > limit:
                        valid = False
                        break
                    bit = bit_of[d]
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if valid:
                    current_row.append(x)
                    if not backtrack(current_row, current_mask | new_mask, x):
                        return False
                    current_row.pop()
            return True
        
        if backtrack([0], 0, 0):
            return results
        else:
            return None
    
    pool = generate_pool(current_max, k, pool_timeout)
    if pool is not None:
        pool.sort(key=lambda x: x[0])
        # Deduplicate by frozenset of diffs (keep smallest max for each diff set)
        seen_diffs = {}
        for max_val, diffs, row_tuple in pool:
            if diffs not in seen_diffs or max_val < seen_diffs[diffs][0]:
                seen_diffs[diffs] = (max_val, row_tuple)
        pool = [(mv, ds, rt) for ds, (mv, rt) in seen_diffs.items()]
        pool.sort(key=lambda x: x[0])
    
    pool_time_elapsed = time.time() - pool_start
    remaining_time = deadline - time.time()

    # Phase 3: Fast local search using pool
    # State: current rows, current diff sets
    cur_rows = rows[:]
    cur_diffs = [row_diffs(r) for r in cur_rows]
    cur_max = current_max
    
    if pool and remaining_time > 10:
        # Build a lookup: for each diff set in pool, quick access
        # We'll iterate through pool sorted by max_val
        
        improved = True
        no_improve = 0
        max_no_improve = 50
        rng = random.Random(42)
        
        while improved and time.time() < deadline - 15:
            improved = False
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            # Find row with largest max
            max_idx = -1
            max_val = -1
            for i, r in enumerate(cur_rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_idx = i
            
            if max_idx == -1:
                break
            
            # Compute union of all other rows' diffs
            others_diffs = set()
            for i in range(n_rows):
                if i != max_idx:
                    others_diffs |= cur_diffs[i]
            
            # Search pool for smallest candidate that fits
            found = False
            for cand_max, cand_diffs, cand_row in pool:
                if cand_max >= max_val:
                    break
                if not (cand_diffs & others_diffs):
                    # Valid swap
                    cur_rows[max_idx] = list(cand_row)
                    cur_diffs[max_idx] = set(cand_diffs)
                    new_max = global_max(cur_rows)
                    if new_max < cur_max:
                        cur_max = new_max
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in cur_rows]
                    improved = True
                    no_improve = 0
                    found = True
                    break
            
            if not found:
                # Try randomized: pick a random row and try to improve it
                i = rng.randint(0, n_rows - 1)
                cur_row_max = cur_rows[i][-1]
                others_diffs = set()
                for j in range(n_rows):
                    if j != i:
                        others_diffs |= cur_diffs[j]
                
                for cand_max, cand_diffs, cand_row in pool:
                    if cand_max >= cur_row_max:
                        break
                    if not (cand_diffs & others_diffs):
                        new_rows = cur_rows[:]
                        new_rows[i] = list(cand_row)
                        new_diffs = cur_diffs[:]
                        new_diffs[i] = set(cand_diffs)
                        new_max = global_max(new_rows)
                        # Accept if it doesn't increase global max
                        if new_max <= cur_max:
                            cur_rows[i] = list(cand_row)
                            cur_diffs[i] = set(cand_diffs)
                            if new_max < cur_max:
                                cur_max = new_max
                            if cur_max < best_max:
                                best_max = cur_max
                                best_rows = [list(r) for r in cur_rows]
                            improved = True
                            no_improve = 0
                        break

    # Phase 4: SA with pool-based and DFS-based candidates
    if time.time() < deadline - 20:
        T = 8.0
        cooling = 0.998
        min_T = 0.1
        
        def dfs_row(row, used_set, cap, row_deadline, randomize=False):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            if len(row) == k + 1:
                return row[:]
            last = row[-1]
            needed = (k + 1) - len(row)
            upper = cap - (needed - 1)
            if upper < last + 1:
                return None
            
            candidates = list(range(last + 1, upper + 1))
            if randomize:
                random.shuffle(candidates)
            else:
                # Try smaller first for compact rows
                candidates.reverse()
            
            for x in candidates:
                if time.time() > row_deadline or time.time() > deadline:
                    return None
                new = {x - a for a in row}
                if new & used_set:
                    continue
                res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize)
                if res is not None:
                    return res
            return None
        
        sa_iter = 0
        while time.time() < deadline - 10:
            if T < min_T:
                break
            sa_iter += 1
            
            # Choose which row to modify
            i = random.randint(0, n_rows - 1)
            
            # Compute used without row i
            used_without_i = set()
            for j in range(n_rows):
                if j != i:
                    used_without_i |= cur_diffs[j]
            
            # Try pool first (fast)
            cand = None
            cur_row_max = cur_rows[i][-1]
            
            # Look for a candidate in pool that is smaller or equal
            search_cap = cur_row_max + random.randint(0, 2)
            for cand_max, cand_diffs, cand_row in pool:
                if cand_max > search_cap:
                    break
                if not (cand_diffs & used_without_i):
                    cand = list(cand_row)
                    break
            
            # If pool didn't yield a better candidate, try DFS
            if cand is None or cand[-1] >= cur_row_max:
                row_deadline = min(deadline - 5, time.time() + 0.3)
                cand = dfs_row([0], used_without_i, search_cap, row_deadline, randomize=True)
            
            if cand is None:
                T *= cooling
                continue
            
            new_diffs = row_diffs(cand)
            # Verify no conflict
            if new_diffs & used_without_i:
                T *= cooling
                continue
            
            # Compute new global max
            new_rows = cur_rows[:]
            new_rows[i] = cand
            new_max = global_max(new_rows)
            delta = new_max - cur_max
            
            if delta <= 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = random.random() < prob
            
            if accept:
                cur_rows[i] = cand
                cur_diffs[i] = new_diffs
                cur_max = new_max
                
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in cur_rows]
            
            T *= cooling

    # Phase 5: Final deterministic polish
    if time.time() < deadline - 5:
        final_rows = best_rows[:]
        final_diffs = [row_diffs(r) for r in final_rows]
        
        for polish_pass in range(5):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(n_rows):
                if time.time() > deadline - 2:
                    break
                cur_last = final_rows[i][-1]
                if cur_last <= 0:
                    continue
                
                used_without = set()
                for j in range(n_rows):
                    if j != i:
                        used_without |= final_diffs[j]
                
                # Try to find a smaller row using DFS
                row_deadline = min(deadline - 2, time.time() + 0.5)
                cand = dfs_row([0], used_without, cur_last - 1, row_deadline, randomize=False)
                if cand is not None:
                    final_rows[i] = cand
                    final_diffs[i] = row_diffs(cand)
                    improved = True
            
            if not improved:
                break
        
        # Check if polish improved things
        final_max = global_max(final_rows)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows[:]

    return best_rows
