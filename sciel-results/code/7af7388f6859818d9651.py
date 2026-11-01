import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    
    # Safety margin: 150s limit, aim for 140s
    deadline = _START + 140.0 

    d = k * (k + 1) // 2  # differences per row
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def rebuild_row(used_other, max_val_hint, rng):
        """
        Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed.
        Uses a greedy approach with some randomization to avoid getting stuck.
        """
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            # Search range: from last_val + 1 to max_val_hint
            # If max_val_hint is None, search up to best_scope + 10
            search_end = max_val_hint if max_val_hint else (best_scope + 20)
            
            # Try a few random starts if greedy fails immediately?
            # No, standard greedy with backtracking is better.
            # But for speed, simple greedy with jitter is good.
            
            # Sort candidates? No, just iterate.
            # To introduce randomness, we can skip some values.
            
            start_x = x + 1
            # Sometimes start a bit higher to avoid local trap?
            # Let's stick to standard greedy first, then perturb.
            
            for trial_x in range(start_x, search_end + 1):
                # Jitter: skip some values to explore
                if rng.random() < 0.05 and trial_x > x + 2:
                    continue
                    
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            
            if not found:
                return None
            x = row[-1]
        
        return row

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    
    # Pre-compute diffs for current best
    best_diffs = [row_diffs_set(r) for r in best_rows]
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    # We will maintain a list of rows and their diffs
    cur_rows = [r[:] for r in best_rows]
    cur_diffs = [row_diffs_set(r) for r in cur_rows]
    cur_scope = best_scope
    
    # Helper to get union of diffs excluding row i
    def get_used_other(i):
        used = set()
        for j in range(n_rows):
            if j != i:
                used.update(cur_diffs[j])
        return used

    steps = 0
    while time.time() < deadline - 5:
        # Pick a row to modify
        r = rng.randrange(n_rows)
        
        # 10% chance to rebuild the row
        if rng.random() < 0.1:
            used_other = get_used_other(r)
            # Try to rebuild with a scope slightly smaller than current if possible, or just current
            # We want to reduce scope, so try to fit in best_scope
            new_row = rebuild_row(used_other, best_scope, rng)
            if new_row is not None:
                new_scope = max(new_row)
                if new_scope <= cur_scope:
                    # Check validity against others (already done in rebuild)
                    cur_rows[r] = new_row
                    cur_diffs[r] = row_diffs_set(new_row)
                    cur_scope = max(max(row) for row in cur_rows)
                    
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [row[:] for row in cur_rows]
                        best_diffs = [row_diffs_set(row) for row in best_rows]
            continue

        # 90% chance to perturb an element
        pos = rng.randint(1, len(cur_rows[r]) - 1)
        old_val = cur_rows[r][pos]
        
        # Determine range for new value
        lo = cur_rows[r][pos - 1] + 1
        hi = cur_rows[r][pos + 1] - 1 if pos < len(cur_rows[r]) - 1 else best_scope + 10
        
        if hi <= lo:
            continue
            
        # Random step size
        step = rng.randint(1, 5)
        if rng.random() < 0.5:
            new_val = old_val + step
        else:
            new_val = old_val - step
            
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue

        # Try the move
        cur_rows[r][pos] = new_val
        new_diffs_r = row_diffs_set(cur_rows[r])
        
        # Check conflicts with other rows
        used_other = get_used_other(r)
        conflicts = len(new_diffs_r & used_other)
        
        # Also check internal conflicts? No, row is strictly increasing, so internal diffs are unique within row?
        # Wait, differences within a row are NOT necessarily unique!
        # Example: 0, 1, 3. Diffs: 1, 3, 2. All unique.
        # Example: 0, 2, 4. Diffs: 2, 4, 2. Conflict!
        # So we MUST check internal conflicts too.
        # My `row_diffs_set` returns a SET, so it hides duplicates.
        # I need a count or a list check.
        
        # Let's re-validate the row strictly.
        is_valid_row = True
        seen_in_row = set()
        row_vals = cur_rows[r]
        for i in range(len(row_vals)):
            for j in range(i+1, len(row_vals)):
                diff = row_vals[j] - row_vals[i]
                if diff in seen_in_row:
                    is_valid_row = False
                    break
                seen_in_row.add(diff)
        if not is_valid_row:
            conflicts += 100 # Heavy penalty for internal conflict
        
        # Total conflicts = conflicts with others + internal issues (penalized)
        # Note: `new_diffs_r` is a set, so it doesn't capture internal duplicates.
        # The `is_valid_row` check handles internal.
        # The `conflicts` variable handles external.
        
        # Calculate Cost
        new_scope = max(max(row) for row in cur_rows)
        
        # We want to minimize Scope.
        # If valid, Cost = Scope.
        # If invalid, Cost = Scope + Penalty * Conflicts.
        
        if conflicts == 0 and is_valid_row:
            cost = new_scope
        else:
            cost = new_scope + 10 * conflicts

        # Compare with current cost
        # Current state might be invalid? We should try to keep it valid if possible.
        # Let's assume current state is valid or "best known valid".
        # Actually, if we start with valid, and only accept valid or better invalid, we might drift.
        # Standard SA: Accept if cost_new < cost_old or random < exp(-(cost_new - cost_old)/T)
        
        # What is cost_old?
        # We need to track if current state is valid.
        # Let's maintain a flag or recompute.
        # For speed, let's assume we only accept moves that keep the row valid internally.
        # If `is_valid_row` is False, reject immediately unless T is very high?
        # Better: Reject internal conflicts always. They are "hard" constraints for the row structure.
        if not is_valid_row:
            cur_rows[r][pos] = old_val
            continue

        # Now only external conflicts matter.
        if conflicts == 0:
            # Valid move
            cur_diffs[r] = new_diffs_r
            cur_scope = new_scope
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [row[:] for row in cur_rows]
                best_diffs = [row_diffs_set(row) for row in best_rows]
        else:
            # Invalid move (external conflict)
            # Calculate delta
            # Old state was valid (assumed), so old_cost = cur_scope
            # New cost = cur_scope + penalty * conflicts
            delta = (new_scope + 10 * conflicts) - cur_scope
            if delta < 0 or rng.random() < (2.71828 ** (-delta / T)):
                cur_diffs[r] = new_diffs_r
                cur_scope = new_scope
            else:
                cur_rows[r][pos] = old_val
                # cur_diffs[r] remains old

        T *= cooling
        if T < min_T:
            T = min_T
            # Restart with higher T if stuck?
            # If no improvement for a while, reset T
            # Simple: just keep going.

    # Final polish
    if time.time() < deadline - 2:
        # Try targeted descent on best_rows
        cur = [r[:] for r in best_rows]
        cur_scope = best_scope
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n_rows):
                for pos in range(1, len(cur[i])):
                    old_val = cur[i][pos]
                    lo = cur[i][pos-1] + 1
                    hi = cur[i][pos+1] - 1 if pos < len(cur[i])-1 else old_val
                    for new_val in range(old_val - 1, lo - 1, -1):
                        cur[i][pos] = new_val
                        # Validate
                        valid = True
                        seen = set()
                        for r_idx in range(n_rows):
                            row = cur[r_idx]
                            for x in range(len(row)):
                                for y in range(x+1, len(row)):
                                    diff = row[y] - row[x]
                                    if diff in seen:
                                        valid = False
                                        break
                                    seen.add(diff)
                            if not valid: break
                        if valid:
                            new_scope = max(max(row) for row in cur)
                            if new_scope < cur_scope:
                                cur_scope = new_scope
                                improved = True
                                break
                            else:
                                cur[i][pos] = old_val
                                break
                        else:
                            cur[i][pos] = old_val
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = cur

    return best_rows
