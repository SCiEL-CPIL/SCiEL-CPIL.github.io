import time
import random
import math
import sys


def _validate(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d in seen:
                    return False
                seen.add(d)
    return True


def _get_scope(rows):
    return max(r[-1] for r in rows)


def _row_diffs(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _row_diffs_mask(row, max_scope):
    """Return bitmask of differences, only up to max_scope."""
    m = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d <= max_scope:
                m |= (1 << d)
    return m


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin

    num_elements = k + 1  # each row has k+1 elements
    num_pairs = num_elements * (num_elements - 1) // 2  # 15 for k=5
    total_diffs_needed = n_rows * num_pairs  # 105 for n_rows=7
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))  # 15 for 105 diffs

    # For (7,5): counting_bound = ceil(sqrt(2*105)) = ceil(sqrt(210)) = 15
    # But we need differences up to the max element, so actual bound is higher.
    # The counting bound on max element: we need n_rows * num_pairs distinct differences,
    # and the max difference is at most the max element M.
    # So we need M(M+1)/2 >= n_rows * num_pairs, giving M >= sqrt(2*n_rows*num_pairs) - 0.5
    # For (7,5): M >= sqrt(2*105) - 0.5 = sqrt(210) - 0.5 ≈ 14.49 - 0.5 = 13.99, so M >= 14
    # But the actual best known is 112, so the counting bound on differences doesn't directly
    # give the element bound. Let me reconsider.
    
    # Actually, the counting bound mentioned in the task is 105, which is different.
    # The number of distinct differences needed is n_rows * num_pairs = 7 * 15 = 105.
    # If the max element is M, then the possible differences range from 1 to M, so we need M >= 105.
    # But that's not right either, because differences can be larger than M... no, differences
    # between elements in a row starting at 0 and ending at M can be at most M.
    # So we need at least 105 distinct positive integers from 1 to M, hence M >= 105.
    # The best known is 112.
    
    # So the counting bound is M >= 105.
    counting_bound_M = total_diffs_needed  # 105

    rng = random.Random(42)

    def _greedy_row(used_mask, max_scope, rng_local):
        """Greedily build a row with smallest possible values."""
        row = [0]
        for _ in range(num_elements - 1):
            found = False
            for val in range(row[-1] + 1, max_scope + 1):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(row):
                    continue
                # Check against used_mask
                conflict = False
                for d in new_diffs:
                    if d < max_scope + 1 and (used_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    row.append(val)
                    found = True
                    break
            if not found:
                return None
        # Update used_mask
        new_mask = 0
        for d in _row_diffs(row):
            if d <= max_scope:
                new_mask |= (1 << d)
        return row, new_mask

    def _backtrack_row(used_mask, max_scope, rng_local, deadline):
        """Backtracking to build a row."""
        def _helper(row_prefix, used_mask_local, deadline):
            if time.time() > deadline:
                return None
            if len(row_prefix) == num_elements:
                new_mask = 0
                for d in _row_diffs(row_prefix):
                    if d <= max_scope:
                        new_mask |= (1 << d)
                return row_prefix, new_mask
            
            row = row_prefix
            last = row[-1]
            
            # Try values in increasing order, but with some randomization
            candidates = list(range(last + 1, max_scope + 1))
            # Add some randomness to break symmetry
            if len(candidates) > 20:
                rng_local.shuffle(candidates[:20])
                candidates = candidates[:20] + candidates[20:]
            
            for val in candidates:
                # Check within-row uniqueness
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(row):
                    continue
                # Check against global used_mask
                conflict = False
                new_mask_add = 0
                for d in new_diffs:
                    if d <= max_scope:
                        if (used_mask_local >> d) & 1:
                            conflict = True
                            break
                        new_mask_add |= (1 << d)
                if conflict:
                    continue
                
                res = _helper(row + [val], used_mask_local | new_mask_add, deadline)
                if res is not None:
                    return res
            return None
        
        return _helper([0], used_mask, deadline)

    def _local_search_improve(rows, max_scope, rng_local, deadline):
        """Try to reduce the scope by local search."""
        cur = [r[:] for r in rows]
        best = [r[:] for r in cur]
        best_scope = _get_scope(cur)
        
        # Precompute masks
        row_masks = [_row_diffs_mask(r, max_scope + 100) for r in cur]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
        
        n = len(cur)
        
        while time.time() < deadline:
            # Pick a row to improve
            row_idx = rng_local.randint(0, n - 1)
            row = cur[row_idx]
            current_scope = row[-1]
            
            if current_scope <= best_scope:
                continue
            
            # Try to reduce the last element
            target_scope = current_scope - rng_local.randint(1, 3)
            if target_scope < num_elements - 1:
                continue
            
            # Get the mask of other rows
            other_mask = global_mask & ~row_masks[row_idx]
            
            # Try to rebuild this row with tighter scope
            new_row, new_mask = _backtrack_row(other_mask, target_scope, rng_local, time.time() + 2)
            if new_row is not None and not (new_mask & other_mask):
                # Check if it's actually better
                new_global_scope = max(max(r[-1] for i, r in enumerate(cur) if i != row_idx), new_row[-1])
                if new_global_scope < best_scope:
                    cur[row_idx] = new_row
                    row_masks[row_idx] = new_mask
                    global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                    best_scope = new_global_scope
                    best = [r[:] for r in cur]
        
        return best, best_scope

    def _element_move_search(rows, max_scope, rng_local, deadline):
        """Try to move individual elements to reduce scope."""
        cur = [r[:] for r in rows]
        best = [r[:] for r in cur]
        best_scope = _get_scope(cur)
        
        row_masks = [_row_diffs_mask(r, max_scope + 100) for r in cur]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
        
        n_rows_cur = len(cur)
        
        while time.time() < deadline:
            row_idx = rng_local.randint(0, n_rows_cur - 1)
            pos = rng_local.randint(1, num_elements - 1)
            row = cur[row_idx]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else best_scope
            if low >= high or low >= old_val:
                continue
            
            new_val = rng_local.randint(low, min(old_val - 1, high))
            if new_val == old_val:
                continue
            
            row[pos] = new_val
            new_diffs = _row_diffs(row)
            
            # Check within-row uniqueness (should be fine since we're just changing one value)
            # Check against other rows
            other_mask = global_mask & ~row_masks[row_idx]
            conflict = False
            for d in new_diffs:
                if d <= max_scope + 100 and (other_mask >> d) & 1:
                    conflict = True
                    break
            
            if not conflict:
                new_mask = 0
                for d in new_diffs:
                    if d <= max_scope + 100:
                        new_mask |= (1 << d)
                
                new_scope = _get_scope(cur)
                if new_scope < best_scope:
                    row_masks[row_idx] = new_mask
                    global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                    best_scope = new_scope
                    best = [r[:] for r in cur]
                else:
                    # Revert
                    row[pos] = old_val
            else:
                row[pos] = old_val
        
        return best, best_scope

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Try backtracking construction at progressively lower scopes
    for scope in range(counting_bound_M + 15, counting_bound_M - 5, -1):
        if time.time() > DEADLINE - 60:
            break
        for seed in range(5):
            if time.time() > DEADLINE - 55:
                break
            rng_local = random.Random(seed * 31 + 7)
            rows = []
            used_mask = 0
            success = True
            for _ in range(n_rows):
                res = _backtrack_row(used_mask, scope, rng_local, time.time() + 5)
                if res is None:
                    success = False
                    break
                row, mask = res
                rows.append(row)
                used_mask |= mask
            
            if success:
                s = _get_scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]
                break  # Move to next scope
    
    # Phase 2: If backtracking failed, use greedy
    if best_rows is None or best_scope > counting_bound_M + 10:
        for seed in range(10):
            if time.time() > DEADLINE - 40:
                break
            rng_local = random.Random(seed * 13 + 3)
            rows = []
            used_mask = 0
            success = True
            for _ in range(n_rows):
                res = _greedy_row(used_mask, 250, rng_local)
                if res is None:
                    success = False
                    break
                row, mask = res
                rows.append(row)
                used_mask |= mask
            
            if success:
                s = _get_scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]

    # Fallback: simple sequential construction
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = set(x - a for a in row)
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _get_scope(rows)

    # Phase 3: Local search to improve
    if time.time() < DEADLINE - 20:
        improved, imp_scope = _local_search_improve(
            best_rows, best_scope, rng, DEADLINE - 15)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 4: Element move search
    if time.time() < DEADLINE - 10:
        improved, imp_scope = _element_move_search(
            best_rows, best_scope, rng, DEADLINE - 5)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Final validation
    if _validate(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = set(x - a for a in row)
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
