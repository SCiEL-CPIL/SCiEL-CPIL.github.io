import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # safety margin before 150s wall-clock
    k1 = k + 1  # number of elements per row
    
    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None):
        """Smallest-next-mark row avoiding used_set."""
        row = [0]
        x = 0
        while len(row) < k1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row_bitset(row, used_mask, cap, row_deadline, diff_masks):
        """
        Find a row with last element <= cap avoiding used_mask.
        row: list of chosen values (starts with [0])
        used_mask: integer bitmask of used differences
        cap: maximum allowed last value
        row_deadline: time limit for this DFS
        diff_masks: precomputed list where diff_masks[x] = bitmask of diffs {x - a for a in row}
                    But since row changes, we compute on the fly instead.
        """
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k1:
            return row[:]
        last = row[-1]
        needed = k1 - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        for x in range(last + 1, upper + 1):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            # Compute new differences
            new_mask = 0
            for a in row:
                d = x - a
                if d > 0:
                    new_mask |= (1 << d)
            if new_mask & used_mask:
                continue
            res = dfs_row_bitset(row + [x], used_mask | new_mask, cap, row_deadline, diff_masks)
            if res is not None:
                return res
        return None

    def compute_diff_mask(row, x):
        """Bitmask of differences x - a for a in row where x > a."""
        mask = 0
        for a in row:
            d = x - a
            if d > 0:
                mask |= (1 << d)
        return mask

    def dfs_row_optimized(row, used_mask, cap, row_deadline):
        """Optimized DFS with early pruning."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k1:
            return row[:]
        last = row[-1]
        needed = k1 - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        
        # Try values in order of increasing "constraint" - smaller gaps first tend to be better
        for x in range(last + 1, upper + 1):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new_mask = 0
            for a in row:
                d = x - a
                if d > 0:
                    new_mask |= (1 << d)
            if new_mask & used_mask:
                continue
            # Additional pruning: check if we can still fit remaining elements
            # Minimum possible next value is x + 1, so we need at least (needed-1) more steps
            res = dfs_row_optimized(row + [x], used_mask | new_mask, cap, row_deadline)
            if res is not None:
                return res
        return None

    # ---- Phase 1: Greedy baseline with multiple random restarts ----
    best_rows = None
    best_max = float('inf')
    
    # Deterministic greedy first
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)
    cur_max = max(r[-1] for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = cur_max
    
    # Randomized greedy restarts to find diverse starting points
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        random.seed(restart * 12345 + 67890)
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            # Randomized greedy: sometimes skip a valid value to explore alternatives
            row = [0]
            x = 0
            while len(row) < k1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    # With some probability, accept this; otherwise try next
                    if random.random() < 0.7:
                        row.append(x)
                        used |= new
                    # else: skip this x, try x+1
            rows.append(row)
            used |= row_diffs(row)
        
        cur_max = max(r[-1] for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
    
    # ---- Phase 2: Difficulty-ordered CSP refinement ----
    # Instead of processing rows in fixed order, we identify which rows are "tightest"
    # (have the most constrained available slots) and refine those first.
    
    if best_rows is not None:
        rows = [r[:] for r in best_rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        
        # Compute "difficulty" for each row: how many valid positions are available
        # A row is more difficult if it has fewer alternative placements
        row_difficulties = []
        for i, row in enumerate(rows):
            # Count how many alternative rows of similar size could fit in its "slot"
            # Simple proxy: the gap between this row's max and the second-largest max
            row_difficulties.append((row[-1], i))
        
        # Sort by decreasing max entry (hardest to shrink first)
        order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        # Iterative improvement: try to shrink each row
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for r in order:
                if time.time() > deadline - 10:
                    break
                cur_last = rows[r][-1]
                if cur_last <= 0:
                    continue
                
                # Try to find a smaller row
                best_cand = None
                # Binary search on cap: find smallest cap where a valid row exists
                lo = 0
                hi = cur_last - 1
                # First check if hi is feasible
                if hi > 0:
                    used_without = set(used)
                    used_without.difference_update(row_diffs(rows[r]))
                    
                    # Try a few cap values from high to low
                    search_caps = []
                    # Try caps close to current first (more likely to succeed)
                    for offset in [0, 1, 2, 5, 10, 20]:
                        c = cur_last - 1 - offset
                        if c > 0:
                            search_caps.append(c)
                    
                    for cap in search_caps:
                        if time.time() > deadline - 10:
                            break
                        row_deadline = min(deadline - 10, time.time() + 3)
                        cand = dfs_row_optimized([0], 0, cap, row_deadline)
                        if cand is not None:
                            # Verify it doesn't conflict with other rows' diffs
                            cand_diffs = row_diffs(cand)
                            if not (cand_diffs & used_without):
                                best_cand = cand
                                break
                        # If DFS found nothing at this cap, try smaller
                        # (but don't go too far down)
                
                if best_cand is not None and best_cand[-1] < cur_last:
                    used.difference_update(row_diffs(rows[r]))
                    rows[r] = best_cand
                    used |= row_diffs(best_cand)
                    improved = True
                    # Update order
                    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        cur_max = max(r[-1] for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
    
    # ---- Phase 3: Final local search with simulated annealing-like swaps ----
    if best_rows is not None and time.time() < deadline - 5:
        rows = [r[:] for r in best_rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        
        # Try swapping pairs of rows to see if we can reduce the max
        n_swaps = 0
        while time.time() < deadline - 5 and n_swaps < 50:
            n_swaps += 1
            # Pick two random rows
            i, j = random.sample(range(n_rows), 2)
            if i == j:
                continue
            
            # Try to rebuild both rows with freed constraints
            freed_diffs = row_diffs(rows[i]) | row_diffs(rows[j])
            used_without = set(used)
            used_without.difference_update(freed_diffs)
            
            # Try to find smaller versions of both rows
            row_i_deadline = min(deadline - 5, time.time() + 2)
            row_j_deadline = min(deadline - 5, time.time() + 2)
            
            # Try row i first with its old cap - 1
            cap_i = rows[i][-1] - 1
            cand_i = dfs_row_optimized([0], 0, cap_i, row_i_deadline) if cap_i > 0 else None
            
            if cand_i is not None:
                cand_i_diffs = row_diffs(cand_i)
                if not (cand_i_diffs & used_without):
                    # Now try row j with updated used set
                    used_temp = used_without | cand_i_diffs
                    cap_j = rows[j][-1] - 1
                    cand_j = dfs_row_optimized([0], 0, cap_j, row_j_deadline) if cap_j > 0 else None
                    
                    if cand_j is not None:
                        cand_j_diffs = row_diffs(cand_j)
                        if not (cand_j_diffs & used_temp):
                            # Both fit!
                            new_max = max(cand_i[-1], cand_j[-1])
                            old_max = max(rows[i][-1], rows[j][-1])
                            if new_max < old_max:
                                used.difference_update(freed_diffs)
                                rows[i] = cand_i
                                rows[j] = cand_j
                                used |= cand_i_diffs | cand_j_diffs
        
        cur_max = max(r[-1] for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
    
    return best_rows
