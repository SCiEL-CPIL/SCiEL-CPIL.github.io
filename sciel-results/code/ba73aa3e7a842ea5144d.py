import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows and k+1 entries per row.
    Uses Simulated Annealing to minimize the scope (largest entry).
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Helper: Greedy initialization
    def greedy_init():
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

    # Helper: Calculate difference counts and conflicts
    def build_state(rows):
        counts = {}
        conflicts = 0
        # We can just count and then sum max(0, c-1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        for c in counts.values():
            if c > 1:
                conflicts += c - 1
        scope = max(max(r) for r in rows)
        return counts, conflicts, scope

    # Initial Solution
    rows = greedy_init()
    counts, conflicts, current_scope = build_state(rows)
    
    best_rows = copy.deepcopy(rows)
    best_scope = current_scope
    best_conflicts = conflicts
    
    # If greedy is already perfect (0 conflicts), we just try to reduce scope
    # If greedy has conflicts (it shouldn't, by definition of greedy construction 
    # where we only pick if no conflict with *existing* used diffs, but wait:
    # Greedy ensures diffs in *current* row don't conflict with *previous* rows' diffs.
    # It does NOT ensure diffs within the *same* row are distinct?
    # "all differences between two entries of the same row are distinct over all rows"
    # My greedy: `if not new_diffs & used`.
    # `used` contains diffs from previous rows.
    # Does it check diffs within the current row?
    # `new_diffs` are diffs of x with previous elements in `row`.
    # If `new_diffs` has duplicates? No, x is increasing, so x-a_i are distinct.
    # If `new_diffs` intersects with diffs already in `row`?
    # The `used` set only contains diffs from *completed* previous rows.
    # It does NOT contain diffs from the *current* row being built.
    # So, `x - a_i` might equal `a_j - a_i`? No, x > a_j.
    # It might equal `a_j - a_p`?
    # Example: row=[0, 3, 6]. Diffs: 3, 6, 3. Duplicate 3!
    # Greedy as written in the prompt:
    # `new = {x - a for a in row}`
    # `if not new & used`
    # It does NOT check if `new` has internal duplicates or if `new` intersects with diffs *already in the current row*.
    # Wait, the prompt's greedy is:
    # `used` is global.
    # It adds `new` to `used`.
    # But it doesn't check if `new` contains a diff that is *already in the current row*.
    # Let's trace: row=[0]. x=1. new={1}. used={}. ok. row=[0,1]. used={1}.
    # x=2. new={2, 1}. used={1}. Intersection {1}. Fail.
    # x=3. new={3, 2}. used={1}. No intersection. ok. row=[0,1,3]. used={1,2,3}.
    # Diffs in row: 1, 3, 2. Distinct.
    # Is it possible to have duplicate diffs within a row in this greedy?
    # If x-a_i == x-a_j -> a_i == a_j (impossible).
    # If x-a_i == a_j - a_p?
    # The check `not new & used` only checks against *previous rows*.
    # It does NOT check against *current row's existing diffs*.
    # So the seed program might produce INVALID rows!
    # "Valid when all positive differences ... are distinct over all rows."
    # If the seed produces duplicates within a row, it's invalid.
    # Let's check the seed logic again.
    # `used` starts empty.
    # Row 1: builds a valid ruler?
    # If I pick x such that `new` doesn't intersect `used`, it doesn't prevent `new` from intersecting with diffs *generated earlier in the same row*.
    # Example: k=2.
    # row=[0].
    # x=1. new={1}. used={}. ok. row=[0,1]. used={1}.
    # x=2. new={2,1}. used={1}. Intersection! Fail.
    # x=3. new={3,2}. used={1}. No intersection. ok. row=[0,1,3].
    # Diffs: 1, 3, 2. All distinct.
    # It seems the greedy *accidentally* produces distinct internal diffs because of the nature of rulers?
    # No. Consider row=[0, 4, 6]. Diffs: 4, 6, 2.
    # How to get [0, 4, 6]?
    # x=4. new={4}. used={}. ok.
    # x=5. new={5,1}. used={4}. ok. row=[0,4,5]? Diffs 4,5,1.
    # x=6. new={6,2,1}. used={4,5,1}. Intersection {1}. Fail.
    # It seems hard to create internal duplicates with this specific greedy because it picks the *smallest* x.
    # However, to be safe, my `build_state` checks all diffs.
    
    # SA Parameters
    T = 10.0
    cooling = 0.999999
    min_T = 0.01
    penalty = 1000
    
    # Pre-calculate max possible scope for array sizing if needed, but dict is fine
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)  # Never touch index 0
        
        old_val = rows[r_idx][j_idx]
        
        # Try a small random step
        # Bias towards decreasing if value is large
        if old_val > current_scope * 0.8:
            delta = random.randint(-5, 2)
        else:
            delta = random.randint(-3, 3)
        
        new_val = old_val + delta
        
        # Check bounds
        if new_val <= 0:
            continue
        if j_idx > 0 and new_val <= rows[r_idx][j_idx - 1]:
            continue
        if j_idx < k and new_val >= rows[r_idx][j_idx + 1]:
            continue
            
        # Calculate changes in conflicts
        # Diffs involving (r_idx, j_idx) and (r_idx, i) for i != j_idx
        row = rows[r_idx]
        
        delta_conflicts = 0
        affected_diffs = []
        
        for i in range(k + 1):
            if i == j_idx:
                continue
            old_d = abs(row[j_idx] - row[i])
            new_d = abs(new_val - row[i])
            
            if old_d == new_d:
                continue
                
            # Update counts tentatively
            # We need to know if old_d was a conflict and new_d is a conflict
            
            # This is tricky to do incrementally without modifying the global `counts` dict 
            # and rolling back if rejected.
            # Let's just modify `counts`, calculate new conflicts, and rollback if needed.
            
            # Optimization: Store old counts for affected diffs
            # But there are k affected diffs.
            
            # Let's do the full update on `counts`
            rollback_info = []
            
            for i in range(k + 1):
                if i == j_idx:
                    continue
                old_d = abs(row[j_idx] - row[i])
                new_d = abs(new_val - row[i])
                
                if old_d == new_d:
                    continue
                
                # Decrement old
                if old_d in counts:
                    counts[old_d] -= 1
                    if counts[old_d] == 0:
                        del counts[old_d]
                rollback_info.append(('dec', old_d))
                
                # Increment new
                if new_d in counts:
                    counts[new_d] += 1
                else:
                    counts[new_d] = 1
                rollback_info.append(('inc', new_d))
            
            # Recalculate conflicts?
            # Expensive to recalculate from scratch O(Scope).
            # We can track `conflicts` variable.
            # For each changed diff d:
            #   old_count = counts[d] + 1 (if we decremented) or counts[d] (if we incremented)
            #   This is getting messy.
            
            # Simpler: Just calculate the new total conflicts from the `counts` dict?
            # Size of counts is ~ Scope. Scope is small (e.g. 200).
            # O(Scope) per move is acceptable if moves are fast.
            
            # Let's just sum the conflicts from the updated `counts` dict.
            new_conflicts = sum(max(0, c - 1) for c in counts.values())
            
            # Calculate new scope
            if new_val > current_scope:
                new_scope = new_val
            elif old_val == current_scope:
                # Need to find new max
                # This is O(nk).
                # We can just check if any other element is larger?
                # Or just take max of the row?
                # Let's just take max of the modified row and compare with others?
                # Actually, if we decreased the max, the new max is either the second largest 
                # in this row or the max of other rows.
                # To keep it simple and fast, let's just calculate max(max(r) for r in rows)
                # only if old_val was the global max.
                new_scope = max(max(r) for r in rows)
            else:
                new_scope = current_scope
            
            delta_scope = new_scope - current_scope
            delta_cost = delta_scope + penalty * (new_conflicts - conflicts)
            
            # Acceptance
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                # Boltzmann probability
                # Avoid overflow
                if delta_cost < 1000: 
                    prob = pow(2.71828, -delta_cost / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Commit change
                rows[r_idx][j_idx] = new_val
                current_scope = new_scope
                conflicts = new_conflicts
                
                if conflicts == 0 and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = copy.deepcopy(rows)
                    
            else:
                # Rollback
                for action, d in reversed(rollback_info):
                    if action == 'dec':
                        # We decremented, so we increment back
                        if d in counts:
                            counts[d] += 1
                        else:
                            counts[d] = 1
                    elif action == 'inc':
                        # We incremented, so we decrement back
                        counts[d] -= 1
                        if counts[d] == 0:
                            del counts[d]
            
        T *= cooling
        if T < min_T:
            T = min_T

    return best_rows

# EVOLVE-BLOCK-END
