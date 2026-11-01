import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then multi-move local search
    with difference-frequency tracking for fast conflict checking.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    # ---- Local search with frequency-based conflict checking ----
    # Key insight: track how many times each difference value is used across all rows.
    # A move is valid if no difference value exceeds count 1 after the move.
    # This allows O(k^2) per move check instead of O(n_rows * k^2).

    def local_search(rows, n_rows, k, deadline):
        """
        Local search minimizing the global max (scope).
        Uses single-element and two-element moves with fast conflict checking.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        row_diffs = [_row_diffs(r) for r in current_rows]
        all_used = _all_used(current_rows)
        
        # Frequency map: diff_value -> count of rows using it
        freq = {}
        for rd in row_diffs:
            for d in rd:
                freq[d] = freq.get(d, 0) + 1
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 300.0
        T_min = 0.1
        cool = 0.999995
        no_improve = 0
        max_no_improve = 50000
        
        def try_move(ri, pi, new_val):
            """Try changing current_rows[ri][pi] to new_val. Returns (success, new_diffs, removed_diffs)."""
            old_val = current_rows[ri][pi]
            if new_val == old_val:
                return False, None, None
            if new_val <= current_rows[ri][pi - 1]:
                return False, None, None
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                return False, None, None
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs for this row
            new_diffs = set()
            n = len(cand_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        return False, None, None  # Internal conflict
                    new_diffs.add(d)
            
            old_diffs = row_diffs[ri]
            
            # Check cross-row conflicts using frequency
            # Removed diffs: old_diffs - new_diffs
            # Added diffs: new_diffs - old_diffs
            removed = old_diffs - new_diffs
            added = new_diffs - old_diffs
            
            # For added diffs, check they don't already exist in other rows
            for d in added:
                if d in freq and freq[d] > 0:
                    # d is used by another row (or same row but we already removed old)
                    # Actually freq includes this row's old usage
                    # We need: freq[d] - (1 if d in old_diffs else 0) + (1 if d in new_diffs else 0) <= 1
                    # Simpler: after removing old_diffs and adding new_diffs, no d should have freq > 1
                    pass
            
            # More careful check:
            # Current freq includes row ri's old_diffs
            # After move: freq' = freq - old_diffs + new_diffs (conceptually)
            # We need all values in freq' to be <= 1
            
            # Check added diffs: for d in added, freq[d] must be 0 or (d is in old_diffs, meaning freq[d] >= 1 but we're removing it)
            # Actually: for d in added:
            #   if d in old_diffs: net change is 0 (removed and re-added), no issue
            #   else: freq[d] must be 0 (no other row uses it)
            for d in added:
                if d not in old_diffs:
                    if freq.get(d, 0) > 0:
                        return False, None, None
            
            # Check removed diffs: for d in removed, if it was only used by this row, that's fine
            # No conflict check needed for removed diffs
            
            return True, new_diffs, (old_diffs, removed, added)

        def apply_move(ri, pi, new_val, new_diffs, old_diffs, removed, added):
            """Apply a validated move."""
            old_val = current_rows[ri][pi]
            current_rows[ri][pi] = new_val
            
            # Update freq
            for d in removed:
                freq[d] -= 1
                if freq[d] == 0:
                    del freq[d]
            for d in added:
                if d not in old_diffs:  # Only truly new diffs
                    freq[d] = freq.get(d, 0) + 1
            
            # Update row_diffs
            row_diffs[ri] = new_diffs
            
            # Update row_maxes
            row_maxes[ri] = max(current_rows[ri])
            
            # Update all_used
            all_used.difference_update(removed)
            all_used.update(added)
            
            return max(row_maxes)

        while time.time() < deadline:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            
            # Generate candidate values
            candidates = []
            
            # Decreasing (primary direction)
            for d in range(1, min(60, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
            
            # Increasing (exploration)
            for d in range(1, min(20, max_allowed - old_val + 1)):
                candidates.append(old_val + d)
            
            # Random jump
            if random.random() < 0.4 and min_allowed <= max_allowed:
                candidates.append(random.randint(min_allowed, max_allowed))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                success, new_diffs, extra = try_move(ri, pi, new_val)
                if not success:
                    continue
                
                old_diffs, removed, added = extra
                new_global_max = max(row_maxes[:ri] + [max(current_rows[ri][:pi] + [new_val] + current_rows[ri][pi+1:])] + row_maxes[ri+1:])
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    current_val = apply_move(ri, pi, new_val, new_diffs, old_diffs, removed, added)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            if no_improve > max_no_improve:
                break
        
        return best_rows, best_val

    # Run multiple trials
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 104729 + 31337
        random.seed(seed)
        
        # Diversified greedy initialization
        init_rows = greedy_construct(n_rows, k)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Local search
        ls_rows, ls_scope = local_search(init_rows, n_rows, k, deadline - 5)
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_solution = ls_rows
    
    # Final polish on best solution
    if time.time() < deadline - 2:
        polish_rows, polish_scope = local_search(
            [r[:] for r in best_solution], n_rows, k, deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

    return best_solution
# EVOLVE-BLOCK-END
