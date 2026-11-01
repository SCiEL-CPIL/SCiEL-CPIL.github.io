import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        row = {0}
        while len(row) < k + 1:
            x = max(row) + 1
            while True:
                new_diffs = {abs(x - a) for a in row}
                if not new_diffs & used:
                    row.add(x)
                    used.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    
    best_rows = [sorted(r) for r in rows]
    best_scope = get_scope(best_rows)
    
    current_rows = [set(r) for r in rows]
    current_used = set(used)
    current_scope = best_scope

    # Helper to check if a move is valid and apply it
    # Returns True if applied
    def apply_move(row_idx, old_val, new_val):
        nonlocal current_used
        if old_val == new_val:
            return True
        row = current_rows[row_idx]
        if new_val in row:
            return False
        
        # Remove old diffs
        diffs_remove = {abs(old_val - v) for v in row if v != old_val}
        for d in diffs_remove:
            if d in current_used:
                current_used.remove(d)
        
        # Check new diffs
        diffs_add = {abs(new_val - v) for v in row if v != old_val}
        if diffs_add & current_used:
            # Revert
            for d in diffs_remove:
                current_used.add(d)
            return False
        
        # Apply
        row.remove(old_val)
        row.add(new_val)
        current_used.update(diffs_add)
        return True

    def revert_move(row_idx, old_val, new_val):
        nonlocal current_used
        if old_val == new_val:
            return
        row = current_rows[row_idx]
        # Remove new diffs
        diffs_remove = {abs(new_val - v) for v in row if v != new_val}
        for d in diffs_remove:
            if d in current_used:
                current_used.remove(d)
        
        # Add old diffs
        # Note: row currently has new_val, not old_val.
        # The set of other elements is the same.
        diffs_add = {abs(old_val - v) for v in row if v != new_val}
        current_used.update(diffs_add)
        
        row.remove(new_val)
        row.add(old_val)

    # Local Search
    while time.time() < deadline:
        # 1. Try to improve current solution
        # Find global max
        max_val = -1
        max_r = -1
        for i, r in enumerate(current_rows):
            m = max(r)
            if m > max_val:
                max_val = m
                max_r = i
        
        # Try to reduce max_val
        row = current_rows[max_r]
        improved = False
        
        # Try decreasing max_val
        # We can try values from max_val - 1 down to 1
        # But limit iterations
        for new_val in range(max_val - 1, 0, -1):
            if time.time() > deadline: break
            if new_val in row: continue
            if apply_move(max_r, max_val, new_val):
                improved = True
                break
        
        if improved:
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [sorted(r) for r in current_rows]
                current_scope = new_scope
            # If scope didn't improve, we might still keep the move if it helps?
            # For now, we only keep if it improves best, or we track current state.
            # Actually, apply_move modifies current state.
            # If it doesn't improve best, we should probably revert to keep current state "good"?
            # Or keep it to explore?
            # Let's keep it.
        
        # 2. Try to reduce other large values?
        # Or try random moves to escape local minima.
        if random.random() < 0.3:
            # Kick
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            if len(row) > 1:
                # Pick a random element (not 0)
                candidates = [v for v in row if v > 0]
                if candidates:
                    old_val = random.choice(candidates)
                    # Try a random new value
                    new_val = random.randint(1, best_scope + 10)
                    if new_val != old_val and new_val not in row:
                         if apply_move(r_idx, old_val, new_val):
                             # Accept kick
                             pass
                         else:
                             # Revert (apply_move already reverts on fail)
                             pass

    return best_rows
