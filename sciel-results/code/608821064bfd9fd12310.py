import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin
    
    num_marks = k + 1
    diffs_per_row = num_marks * (num_marks - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Initial guess for max value
    # Simple counting bound is roughly sqrt(2 * total_diffs)
    # For n_rows=4, k=7 (8 marks), diffs_per_row = 28, total = 112. sqrt(224) ~ 15.
    # But differences are not just count, they are values.
    # Known best for (4,7) is 134.
    # Let's start with a generous bound to ensure validity, then shrink.
    # Or start with a tight bound and grow?
    # Growing is safer for validity, but we want to minimize.
    # Let's initialize with a valid solution using greedy, then anneal to shrink.
    
    def get_row_diffs(row):
        # row is a numpy array of shape (num_marks,)
        # Returns a set of differences
        # Efficient way:
        return set(int(row[j] - row[i]) for i in range(num_marks) for j in range(i+1, num_marks))
        
    def greedy_row(used, num_marks, max_val, seed=None):
        if seed is not None:
            random.seed(seed)
        row = [0]
        x = 0
        while len(row) < num_marks:
            found = False
            candidates = list(range(x+1, min(x+1+200, max_val+1)))
            if seed is not None:
                random.shuffle(candidates)
            else:
                candidates.sort()
            for c in candidates:
                new_diffs = {c - a for a in row}
                if not new_diffs & used:
                    row.append(c)
                    used = used | new_diffs
                    x = c
                    found = True
                    break
            if not found:
                return None
        return row

    # Phase 1: Get a valid starting point
    # Try to find a valid configuration with a relatively small max_val
    # Greedy tends to produce large values. Let's try a few seeds.
    best_rows = None
    best_scope = float('inf')
    
    # Try deterministic greedy first with a high limit to ensure we get *a* solution
    used = set()
    rows = []
    success = True
    for _ in range(n_rows):
        row = greedy_row(used, num_marks, 500, seed=None)
        if row is None:
            success = False
            break
        rows.append(row)
        used |= get_row_diffs(row)
    
    if success:
        best_rows = [np.array(r, dtype=np.int32) for r in rows]
        best_scope = max(max(r) for r in rows)
        
    # Try a few randomized greedies to see if we can get a better start
    for trial in range(20):
        if time.time() > deadline - 10: break
        random.seed(100 + trial)
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = greedy_row(used, num_marks, 300, seed=trial)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= get_row_diffs(row)
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [np.array(r, dtype=np.int32) for r in rows]

    if best_rows is None:
        # Fallback: return invalid or a very loose valid one?
        # The task says "Valid when...". If we can't find one, we return something.
        # But greedy with 500 should always work for small n_rows, k.
        return [[0]] * n_rows

    # Phase 2: Simulated Annealing to minimize the maximum entry
    # State: list of rows (numpy arrays)
    # Objective: Minimize max(max(row) for row in rows) + Penalty * (Number of Conflicts)
    # Conflict: Any difference appearing in more than one row.
    
    current_rows = [r.copy() for r in best_rows]
    current_max = best_scope
    
    def calculate_penalty(rows):
        """Returns (conflict_count, max_entry)"""
        used = set()
        conflicts = 0
        max_entry = 0
        for row in rows:
            diffs = get_row_diffs(row)
            for d in diffs:
                if d in used:
                    conflicts += 1
            used.update(diffs)
            max_entry = max(max_entry, max(row))
        return conflicts, max_entry

    current_conflicts, current_max = calculate_penalty(current_rows)
    # Note: Greedy ensures current_conflicts == 0 initially.
    
    best_rows_sa = [r.copy() for r in current_rows]
    best_score_sa = current_max # Since conflicts are 0, score is just max
    
    T = 50.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    steps_per_temp = 100
    
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    
    while T > T_min and time.time() < deadline - 5:
        # Perform a step
        # Pick a random row
        r_idx = rng.integers(0, n_rows)
        # Pick a random index in that row to perturb (not the first one, which is 0)
        # Actually, perturbing one element breaks increasing order.
        # Better: Pick a random row, and try to shift a suffix or swap?
        # Simplest valid move for "increasing integers":
        # Pick a random row, pick a random position i (1..num_marks-1).
        # Try to change row[i] by +/- delta.
        # But this might violate strict increase.
        
        # Alternative Move: 
        # 1. Pick a row.
        # 2. Pick a "segment" of the row to regenerate? Too expensive.
        # 3. Local search move: Pick a random row, pick a random element (not first), 
        #    try to replace it with a nearby integer that maintains order and reduces conflicts.
        
        row = current_rows[r_idx]
        # Choose a position to perturb. Avoid position 0.
        pos = rng.integers(1, num_marks)
        old_val = row[pos]
        
        # Generate a candidate new value
        # We want to explore the space.
        # If we just do +/- 1, it's very slow.
        # Let's try a random jump within a window.
        # Window size depends on T? Or fixed?
        # Let's try a random value in [old_val - 10, old_val + 10]
        # But must maintain: row[pos-1] < new_val < row[pos+1] (if pos < num_marks-1)
        
        lower = row[pos-1] + 1
        upper = row[num_marks-1] if pos == num_marks-1 else row[pos+1] - 1
        
        if lower > upper:
            continue
            
        # Pick a new value
        # With some probability, try to reduce the value (to shrink scope)
        # With some probability, try a random value in range
        if rng.random() < 0.5:
            # Try to reduce
            # Pick a value between lower and old_val
            if lower < old_val:
                new_val = rng.integers(lower, old_val + 1)
            else:
                new_val = old_val
        else:
            # Random explore
            # Pick from [lower, upper]
            # But limit the range to avoid huge jumps that break other constraints?
            # Actually, huge jumps might be good to escape local minima.
            # Let's pick from [lower, min(upper, old_val + 50)]
            new_val = rng.integers(lower, min(upper, old_val + 50) + 1)
            
        if new_val == old_val:
            continue
            
        # Calculate delta in conflicts and max
        # We need to know which diffs are affected.
        # Diffs involving index pos:
        # (row[pos] - row[i]) for i < pos
        # (row[j] - row[pos]) for j > pos
        
        # This is complex to calculate incrementally without storing diff counts.
        # Given the small size (num_marks <= 8, n_rows <= 4), 
        # full recalculation of penalty is O(n_rows * num_marks^2) = O(4 * 64) = 256 ops.
        # This is very fast. We can just recalculate.
        
        # Apply change
        row[pos] = new_val
        
        new_conflicts, new_max = calculate_penalty(current_rows)
        new_score = new_max + new_conflicts * 1000 # Heavy penalty for conflicts
        
        old_score = current_max + current_conflicts * 1000
        
        delta_score = new_score - old_score
        
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            # Accept
            current_conflicts = new_conflicts
            current_max = new_max
            
            if new_conflicts == 0 and new_max < best_score_sa:
                best_score_sa = new_max
                best_rows_sa = [r.copy() for r in current_rows]
        else:
            # Reject
            row[pos] = old_val
            
        T *= alpha
        
    # Final check
    if best_score_sa < best_scope:
        best_rows = best_rows_sa
        best_scope = best_score_sa
        
    # Convert back to list of lists
    return [list(r) for r in best_rows]

