import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # 1. Baseline: Greedy construction
    # This ensures we have a valid starting point.
    baseline = _greedy_construct(n_rows, k)
    best_rows = [row[:] for row in baseline]
    best_score = _max_entry(baseline)
    
    if n_rows <= 2 and k <= 3:
        return best_rows

    # 2. Simulated Annealing
    # State: list of rows
    # Cost: W_CONFLICT * (num_conflicting_diffs) + max_entry
    
    W_CONFLICT = 1000 # Weight for conflicts, ensures validity is prioritized
    
    # Initialize SA
    current_rows = [row[:] for row in baseline]
    
    # Precompute differences for current state to enable fast updates
    # We maintain:
    # - diff_counts: dictionary {diff_value: count}
    # - row_diffs: list of sets of differences for each row
    
    row_diffs = []
    diff_counts = {}
    
    for row in current_rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diffs.add(d)
                diff_counts[d] = diff_counts.get(d, 0) + 1
        row_diffs.append(diffs)
    
    # Calculate initial cost
    num_conflicts = _calc_conflicts(diff_counts)
    current_cost = W_CONFLICT * num_conflicts + _max_entry(current_rows)
    
    # SA Parameters
    T = 50.0
    min_T = 0.1
    cooling_rate = 0.995
    steps_per_temp = 20
    steps = 0
    total_steps = 0
    
    # We want to explore lower scopes.
    # The greedy solution is valid (conflicts=0).
    # We will start with T high enough to allow moves that might increase scope
    # but reduce conflicts (if we started invalid) or just move around in valid space
    # but we start valid. So moves that increase conflicts are expensive.
    # Moves that keep conflicts 0 but reduce scope are good.
    # Moves that keep conflicts 0 but increase scope are bad (unless they allow further reduction later).
    
    # To encourage scope reduction, we can bias the move generation or cost function.
    # Current cost: W * K + S.
    # If K=0, cost = S. We minimize S.
    # If we move to a state with K>0, cost jumps by W.
    # So we mostly stay in valid states unless T is very high.
    
    while time.time() < deadline - 5.0:
        for _ in range(steps_per_temp):
            # Check time frequently
            if time.time() > deadline - 5.0:
                break
            
            # Select a row and an index to perturb
            r_idx = random.randint(0, n_rows - 1)
            # Indices 1..k are variable. Index 0 is fixed at 0.
            j_idx = random.randint(1, k)
            
            old_val = current_rows[r_idx][j_idx]
            
            # Determine valid range for new value
            # Must be > previous and < next (if exists)
            low_bound = current_rows[r_idx][j_idx - 1] + 1
            if j_idx < k:
                high_bound = current_rows[r_idx][j_idx + 1] - 1
            else:
                # For the last element, we can go higher, but let's limit it to current max + some delta
                # Or just allow a range around current value
                high_bound = old_val + 20 
            
            if high_bound < low_bound:
                # No valid move in this neighborhood? 
                # This can happen if the sequence is tight (e.g., 0,1,2,3,4).
                # In that case, we cannot change this element without moving others.
                # For simplicity, skip this move.
                continue
                
            # Choose a new value
            # To encourage exploration towards smaller values (lower scope),
            # we can bias the choice towards the lower bound, but allow higher.
            # Uniform random in [low, high] is okay.
            new_val = random.randint(low_bound, high_bound)
            
            if new_val == old_val:
                continue
                
            # Perform Move: Update diff_counts and row_diffs
            # 1. Remove old diffs for row r_idx
            old_diffs_set = row_diffs[r_idx]
            for d in old_diffs_set:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # 2. Apply change to row
            current_rows[r_idx][j_idx] = new_val
            
            # 3. Calculate new diffs for row r_idx
            new_diffs_set = set()
            row = current_rows[r_idx]
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    new_diffs_set.add(d)
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            
            row_diffs[r_idx] = new_diffs_set
            
            # 4. Calculate new cost
            new_num_conflicts = _calc_conflicts(diff_counts)
            new_max_entry = _max_entry(current_rows)
            new_cost = W_CONFLICT * new_num_conflicts + new_max_entry
            
            # 5. Accept/Reject
            delta_cost = new_cost - current_cost
            
            accepted = False
            if delta_cost < 0:
                accepted = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                current_cost = new_cost
                # If valid and better, update best
                if new_num_conflicts == 0:
                    if new_max_entry < best_score:
                        best_score = new_max_entry
                        best_rows = [row[:] for row in current_rows]
                        # Optional: early stop if we hit theoretical lower bound?
                        # Simple counting bound is 105.
                        if best_score <= 105:
                            return best_rows
            else:
                # Revert move
                current_rows[r_idx][j_idx] = old_val
                
                # Revert diff_counts and row_diffs
                # We need to subtract the new_diffs we just added
                for d in new_diffs_set:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                # We don't need to re-add old_diffs to diff_counts explicitly if we just 
                # "undo" the addition? 
                # Wait. 
                # Step 1 removed old.
                # Step 3 added new.
                # If rejected, we want state to be: old added, new removed.
                # So we must add old_diffs back to diff_counts.
                for d in old_diffs_set:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                
                row_diffs[r_idx] = old_diffs_set

            total_steps += 1
            
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_rows

def _calc_conflicts(diff_counts):
    """Calculate number of duplicate differences.
    If a diff appears C times, it contributes max(0, C-1) to conflicts.
    """
    conflicts = 0
    for c in diff_counts.values():
        if c > 1:
            conflicts += c - 1
    return conflicts

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    m = 0
    for row in rows:
        if row:
            val = row[-1]
            if val > m:
                m = val
    return m
