import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: we need total_diffs distinct positive integers 1..M
    # So M >= total_diffs. Since max entry is M, and differences go up to M,
    # actually the max difference is the max entry (since min is 0).
    # So max_entry >= total_diffs.
    lb = total_diffs

    def get_diff_mask(row):
        """Compute the bitset mask of differences for a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                mask |= (1 << d)
        return mask

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen_mask = 0
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            mask = get_diff_mask(row)
            if seen_mask & mask:
                return False
            seen_mask |= mask
        return True

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- Fast Greedy Initial Solution ----
    def greedy_initial(rng, cap=200):
        used_mask = 0
        rows = []
        for r in range(n_rows):
            row = [0]
            cur_mask = 0
            x = 1
            while len(row) < m and x < cap:
                new_d_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    new_d_mask |= (1 << d)
                if (used_mask | cur_mask) & new_d_mask:
                    x += 1
                    continue
                # Accept
                row.append(x)
                cur_mask |= new_d_mask
                x += 1
            if len(row) < m:
                return None
            rows.append(row)
            used_mask |= cur_mask
        return rows

    # ---- Backtracking Row Generation with Bitset Pruning ----
    def backtrack_row(used_mask, k_val, max_val, rng, cap=300):
        """
        Generate one row of k_val+1 entries starting at 0, max entry <= max_val.
        Avoids used_mask.
        Returns (row, new_mask) or (None, 0).
        """
        row = [0]
        cur_mask = 0
        
        def dfs(pos, last_val):
            nonlocal row, cur_mask
            if time.time() > deadline - 5:
                return False
            
            if pos == k_val:
                return True
            
            # Determine range for next value
            # Must be > last_val and <= max_val
            # Also need space for remaining elements
            rem = k_val - pos - 1
            min_next = last_val + 1
            max_next = max_val - rem  # ensure we can fit remaining elements strictly increasing
            
            if min_next > max_next:
                return False
            
            # Collect candidates
            candidates = []
            # To speed up, we can sample or iterate. 
            # For small k (7), iterating is fine.
            # We prefer smaller values to keep scope tight, but need randomness.
            # We will iterate from min_next to max_next, but collect valid ones.
            
            # Optimization: If the range is large, we might want to randomize start or step.
            # But for k=7, max_val ~ 134, range is small enough.
            
            for x in range(min_next, max_next + 1):
                new_d_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    # If d > max_val, it's a conflict with the scope constraint? 
                    # No, the difference itself just needs to be unique. 
                    # But if d > max_val, it's fine as long as it's not used.
                    # However, if we are constraining the *entries* to max_val, 
                    # differences can be up to max_val.
                    new_d_mask |= (1 << d)
                
                if (used_mask | cur_mask) & new_d_mask:
                    continue
                
                # Check if this x allows completion? (Simple heuristic: just try)
                candidates.append((x, new_d_mask))
                
                # Limit candidates to avoid explosion if range is huge (not expected here)
                if len(candidates) > 50:
                    break

            if not candidates:
                return False
            
            # Shuffle for randomness, but bias towards smaller
            # Simple approach: sort by x, then take a random subset or shuffle
            candidates.sort(key=lambda t: t[0])
            
            # With high probability, try the smallest few. Otherwise random.
            if rng.random() < 0.7:
                # Try smallest 5
                trial_list = candidates[:5]
            else:
                # Random selection
                trial_list = rng.sample(candidates, min(10, len(candidates)))
            
            for x, new_mask in trial_list:
                old_row_len = len(row)
                row.append(x)
                old_cur_mask = cur_mask
                cur_mask |= new_mask
                
                if dfs(pos + 1, x):
                    return True
                else:
                    row.pop()
                    cur_mask = old_cur_mask
            
            return False
        
        if dfs(0, 0):
            return row, cur_mask
        return None, 0

    # ---- Simulated Annealing with Row Regeneration ----
    def anneal(rows, rng, max_iter=20000):
        rows = [list(r) for r in rows]
        used_mask = 0
        row_masks = []
        for r in rows:
            mask = get_diff_mask(r)
            row_masks.append(mask)
            used_mask |= mask
        
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]
        
        T = 5.0
        cooling = 0.999
        min_T = 0.2

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
                
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_mask = row_masks[r_idx]
            
            # Remove old diffs from global mask
            used_mask &= ~old_mask
            
            # Generate new row
            # Try to keep scope similar or smaller
            # Use a cap based on current best or current max
            cap = int(best_max * 1.1) + 10
            new_row, new_mask = backtrack_row(used_mask, k, cap, rng)
            
            if new_row is None:
                # Restore
                used_mask |= old_mask
                row_masks[r_idx] = old_mask
                continue
            
            new_global = max(max(rows[i][-1] for i in range(n_rows) if i != r_idx), new_row[-1])
            delta = new_global - current_max
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    accept = True
            
            if accept:
                rows[r_idx] = new_row
                row_masks[r_idx] = new_mask
                used_mask |= new_mask
                current_max = new_global
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                # Restore
                used_mask |= old_mask
                row_masks[r_idx] = old_mask
            
            T = max(min_T, T * cooling)
            
        return best_rows

    # ---- Scope Reduction Loop ----
    def reduce_scope(rows, rng, current_scope, max_attempts=20):
        """
        Try to find a solution with scope < current_scope.
        """
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        target = best_scope - 1
        
        if target < lb:
            return best_rows, best_scope
            
        for _ in range(max_attempts):
            if time.time() > deadline - 10:
                break
            if target < lb:
                break
            
            # Try to construct a solution with max entry <= target
            # We do this by regenerating rows one by one, forcing the constraint
            new_rows = []
            used_mask = 0
            success = True
            
            # Sort rows by their current max value descending, so we tackle the tightest constraints first?
            # Or just random order.
            indices = list(range(n_rows))
            rng.shuffle(indices)
            
            for r_idx in indices:
                # The new row must have max <= target
                new_row, new_mask = backtrack_row(used_mask, k, target, rng)
                if new_row is None:
                    success = False
                    break
                new_rows.append(new_row)
                used_mask |= new_mask
            
            if success:
                # Verify validity and scope
                if is_valid(new_rows):
                    new_scope = scope_of(new_rows)
                    if new_scope < best_scope:
                        best_rows = new_rows
                        best_scope = new_scope
                        target = best_scope - 1
                        # Restart loop with new target
                        return best_rows, best_scope
                    else:
                        # It was valid but didn't improve scope? (Shouldn't happen if target < current)
                        pass
                        
            # If failed, try a slightly relaxed target or just retry with randomization
            # To avoid getting stuck, we can try target again with different random seed
            
        return best_rows, best_scope

    # ---- Local Element Refinement ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        used_mask = 0
        row_masks = []
        for r in rows:
            mask = get_diff_mask(r)
            row_masks.append(mask)
            used_mask |= mask
            
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            
            old_val = row[pos]
            old_mask = row_masks[r_idx]
            
            # Try to move element down or up slightly
            # Focus on moving down to reduce scope
            for delta in [-1, -2, 1, 2]:
                new_val = old_val + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                    
                test_row = row[:]
                test_row[pos] = new_val
                
                # Check internal validity
                new_mask = 0
                ok = True
                n = len(test_row)
                for i in range(n):
                    ri = test_row[i]
                    for j in range(i + 1, n):
                        d = test_row[j] - ri
                        bit = 1 << d
                        if new_mask & bit:
                            ok = False
                            break
                        new_mask |= bit
                    if not ok:
                        break
                        
                if not ok:
                    continue
                
                # Check cross-row validity
                # Remove old row mask from used, check conflict, add new
                global_without_old = used_mask & ~old_mask
                if global_without_old & new_mask:
                    continue
                    
                # Valid move
                other_max = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row[-1])
                
                if new_scope <= current_max:
                    # Accept
                    rows[r_idx] = test_row
                    row_masks[r_idx] = new_mask
                    used_mask = global_without_old | new_mask
                    current_max = new_scope
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break  # One change per iteration
                    
        return best_rows

    # ---- Main Search ----
    rng = random.Random(42)
    
    # 1. Initial Solution
    best_rows = greedy_initial(rng, cap=200)
    if best_rows is None or not is_valid(best_rows):
        # Fallback: very generous cap
        best_rows = greedy_initial(rng, cap=500)
        if best_rows is None:
            # Absolute fallback: simple greedy without bitset speedup (should not happen)
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < m:
                    new_d = {x - a for a in row}
                    if not (new_d & used):
                        row.append(x)
                        used |= new_d
                    x += 1
                best_rows.append(row)
                
    best_scope = scope_of(best_rows)
    
    # 2. Annealing
    rows = anneal(best_rows, rng, max_iter=15000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
            
    # 3. Local Refinement
    rows = local_refine(best_rows, rng, max_iter=5000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
            
    # 4. Scope Reduction (Targeted Backtracking)
    # This is the key structural addition from the inspiration
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        rng2 = random.Random(1000 + trial * 137)
        rows, new_scope = reduce_scope(best_rows, rng2, best_scope, max_attempts=10)
        if new_scope < best_scope:
            best_rows = rows
            best_scope = new_scope
            
            # Refine the new better solution
            rows = local_refine(best_rows, rng2, max_iter=3000)
            if is_valid(rows):
                s = scope_of(rows)
                if s < best_scope:
                    best_rows = rows
                    best_scope = s
                    
            # Anneal the new better solution
            rows = anneal(best_rows, rng2, max_iter=5000)
            if is_valid(rows):
                s = scope_of(rows)
                if s < best_scope:
                    best_rows = rows
                    best_scope = s
                    
    # 5. Final Polish
    best_rows = local_refine(best_rows, rng, max_iter=2000)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)
        
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
