import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 115.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2

    def row_diffs_bitset(row):
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = 0
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs_bitset(r)
            if d & used:
                return False
            # Check internal distinctness (should be guaranteed by construction if we use bitset and check popcount? 
            # No, bitset just stores existence. We need to ensure the row itself has distinct diffs.
            # A row of length L has L*(L-1)/2 diffs. If the bitset has that many bits set, they are distinct.
            if bin(d).count("1") != needed:
                return False
            used |= d
        return True

    # ---- 1. Deterministic Greedy (Baseline) ----
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    if d <= 0:
                        continue
                    if (used >> d) & 1:
                        valid = False
                        break
                    if (new_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                if valid:
                    row.append(x)
                    used |= new_mask
            rows.append(row)
        return rows

    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    
    # ---- 2. Memoized Randomized DFS for Decision Problem ----
    # Try to find a valid set with max <= M
    
    def dfs_search(M, time_limit):
        start_t = time.time()
        # Memoization: (row_idx, used_mask) -> False (failed)
        # We don't memoize True because we just need one solution.
        failed_states = set()
        
        def dfs(row_idx, used_mask, current_rows):
            if time.time() - start_t > time_limit:
                return None
            
            if row_idx == n_rows:
                return [list(r) for r in current_rows]
            
            state_key = (row_idx, used_mask)
            if state_key in failed_states:
                return None
            
            # Generate candidates for the next row
            # We want a row of k+1 elements, starting at 0, max <= M
            # And diffs must not conflict with used_mask
            
            # Strategy: Randomized Greedy construction of the row
            # To increase diversity, we can shuffle the "candidate selection"
            
            # We'll try to build the row element by element
            # To avoid deep recursion for row construction, we do it iteratively
            # But we need to backtrack if a choice leads to a dead end.
            
            # Simplified: Try a few random "greedy" rows
            
            # Heuristic: The row should be "dense" to fit in M
            # Let's try to construct a row using a local search or just random valid picks
            
            # For speed, let's just try the "standard" greedy row construction 
            # but with some randomization in candidate order
            
            # Actually, a better approach:
            # Pre-compute a list of "valid row templates" for max M?
            # Too expensive.
            
            # Let's use a recursive helper for row construction
            def build_row(current_row, current_mask):
                if len(current_row) == k + 1:
                    return current_row[:]
                
                # If time is tight, prune
                if time.time() - start_t > time_limit:
                    return None
                
                # Determine next element candidates
                # Must be > current_row[-1] and <= M
                # And diffs with existing elements must not be in used_mask or current_mask
                
                # To add randomness, we can pick from the top W candidates
                # But for now, let's try smallest first, then random restarts handle diversity
                
                # Optimization: 
                # If current_row[-1] + (k+1 - len(current_row)) > M, prune
                slots_left = (k + 1) - len(current_row)
                if current_row[-1] + slots_left > M:
                    return None
                
                next_start = current_row[-1] + 1
                # We can't iterate up to M if M is large and we have few slots?
                # But M is small (~100).
                
                # To allow backtracking, we need to try multiple candidates
                # We'll try candidates in increasing order, but with a random skip?
                
                # Let's just try the smallest valid candidate first.
                # If it fails, the caller (dfs) will try another row strategy?
                # No, this is deterministic.
                
                # To make it randomized, we should shuffle the candidate order.
                # But generating all candidates up to M is O(M).
                
                # Let's try a window of candidates
                cands = []
                for x in range(next_start, M + 1):
                    new_mask = 0
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d <= 0: continue
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        if (current_mask >> d) & 1:
                            valid = False
                            break
                        new_mask |= (1 << d)
                    if valid:
                        cands.append((x, new_mask))
                    if len(cands) > 20: # Limit candidates for speed
                        break
                        # Note: This limits the search to "small" next steps.
                        # If we need a big jump, we might miss it.
                        # But for minimizing max, small steps are preferred.
                
                # Randomize the order of candidates
                random.shuffle(cands)
                
                for x, new_mask in cands:
                    res = build_row(current_row + [x], current_mask | new_mask)
                    if res is not None:
                        return res
                
                return None
            
            # Start building the row
            # Initial row [0], mask 0
            # Note: We must ensure the row itself is valid (distinct diffs)
            # The build_row function handles this via current_mask
            
            # To add diversity between rows, we can vary the "randomness"
            # But for a specific M, we just need one solution.
            
            # Let's try a few different "strategies" for the first element?
            # No, it's always 0.
            
            # We'll just call build_row.
            # If it returns None, we mark state as failed.
            
            # To add randomness to the DFS itself, we can randomly choose 
            # between "Greedy" (smallest first) and "Random" (shuffled)
            # But build_row already shuffles.
            
            row = build_row([0], 0)
            
            if row is None:
                failed_states.add(state_key)
                return None
            
            # Recurse to next row
            return dfs(row_idx + 1, used_mask | row_diffs_bitset(row), current_rows + [row])
        
        # Run DFS
        # We might need to restart with different random seeds if the first fails
        # But the shuffling inside build_row provides some diversity.
        
        return dfs(0, 0, [])

    # ---- 3. Local Search (Simulated Annealing) ----
    def local_search(start, rng, steps, T0=3.0):
        cur = [list(r) for r in start]
        # Pre-compute diff masks
        row_masks = [row_diffs_bitset(r) for r in cur]
        cur_used = 0
        for m in row_masks:
            cur_used |= m
            
        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum
        
        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        
        for it in range(steps):
            if it % 100 == 0 and remaining() < 10:
                break
            
            r = rng.randrange(n_rows)
            row = cur[r]
            
            # Move: Change one element
            p = rng.randrange(1, len(row))
            lo = row[p - 1] + 1
            hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 10)
            if lo > hi:
                continue
            
            # Bias towards smaller values
            v = lo + int((hi - lo + 1) * (rng.random() ** 1.5))
            if v == row[p]:
                continue
            
            new_row = row[:]
            new_row[p] = v
            if not all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            nd = row_diffs_bitset(new_row)
            if bin(nd).count("1") != needed:
                continue
            
            other = cur_used & ~(1 << 64) # Mask out other bits? No, just subtract
            # other = cur_used - row_masks[r] is not set difference of bitsets
            # We need to clear the bits of row_masks[r] from cur_used
            other_mask = cur_used & ~row_masks[r]
            
            if nd & other_mask:
                continue
            
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            if d <= 0 or rng.random() < math.exp(-d / T):
                cur[r] = new_row
                row_masks[r] = nd
                cur_used = other_mask | nd
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            T = max(0.05, T * 0.9995)
        
        return best

    # ---- Main Search Loop ----
    rng = random.Random(42)
    
    # Phase 1: Try to lower the scope using DFS
    # Start from best_scope - 1
    target_scope = best_scope - 1
    
    # Allocate time for DFS
    # If we have 115s total, let's give 50s to DFS, 60s to Local Search
    
    dfs_budget = 50.0
    ls_budget = 55.0
    
    while target_scope > 50 and (time.time() - t0) < dfs_budget:
        # Try to find a solution with max <= target_scope
        # Run DFS with a time limit
        limit = min(5.0, dfs_budget - (time.time() - t0))
        if limit < 1.0:
            break
            
        res = dfs_search(target_scope, limit)
        
        if res is not None:
            # Found a valid solution with max <= target_scope
            # The actual max might be less than target_scope
            actual_scope = scope_of(res)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = res
            # Try to go even lower
            target_scope = actual_scope - 1
        else:
            # Failed to find solution for this target
            # Try a different seed? Or just move down?
            # Moving down is safer.
            target_scope -= 1
            
    # Phase 2: Local Search to polish
    if remaining() > 5:
        # Run multiple local searches
        steps = 5000
        for i in range(10):
            if remaining() < 5:
                break
            if i % 2 == 0:
                res = local_search(best_rows, rng, steps)
            else:
                # Perturb
                pert = [list(r) for r in best_rows]
                for _ in range(2):
                    rr = rng.randrange(n_rows)
                    pp = rng.randrange(1, len(pert[rr]))
                    lo = pert[rr][pp - 1] + 1
                    hi = (pert[rr][pp + 1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 20
                    if lo <= hi:
                        nv = rng.randrange(lo, hi + 1)
                        new_row = pert[rr][:]
                        new_row[pp] = nv
                        if all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                            nd = row_diffs_bitset(new_row)
                            if bin(nd).count("1") == needed:
                                other = cur_used & ~row_masks[rr] # Re-calc needed? 
                                # Just check validity simply
                                used = 0
                                valid = True
                                for j, row in enumerate(pert):
                                    d = nd if j == rr else row_diffs_bitset(row)
                                    if d & used:
                                        valid = False
                                        break
                                    used |= d
                                if valid:
                                    pert[rr] = new_row
                                    break
                res = local_search(pert, rng, steps)
            
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final Safety
    if not is_valid(best_rows):
        best_rows = base
        
    return best_rows
