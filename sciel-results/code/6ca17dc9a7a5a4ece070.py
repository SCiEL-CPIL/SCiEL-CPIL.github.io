"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    # 1. Initialization: Standard Greedy to get a valid baseline
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
    
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    def get_diffs(row):
        return set(row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1))
    
    # 2. Local Search / Simulated Annealing
    # State: list of rows (lists of ints)
    # Objective: Minimize max entry, subject to all diffs being distinct.
    # Moves:
    #   - Increase a value (repair)
    #   - Decrease a value (compress) if valid
    #   - Swap values in a row (if order preserved) - actually, just random walk on values is better.
    
    current_rows = [r[:] for r in best_rows]
    current_score = best_score
    
    # Temperature schedule
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random row and a random position (1..k)
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k)
        
        # Generate a neighbor
        # Bias towards decreasing values (to improve score) if possible, or small random steps
        # Let's try a random step: -2, -1, 0, 1, 2
        step = random.randint(-2, 2)
        if step == 0:
            continue
            
        new_val = current_rows[r_idx][p_idx] + step
        
        # Check validity constraints for this row:
        # 1. Strictly increasing: prev < new_val < next
        # 2. Diffs involving new_val must not collide with existing diffs in this row (except the ones being replaced)
        # 3. Diffs involving new_val must not collide with diffs in other rows.
        
        # Optimization: We can't easily check "collisions with other rows" without a global set.
        # Strategy: 
        # 1. Tentatively update the row.
        # 2. Check row internal validity.
        # 3. Check global validity.
        # To make global check fast, we maintain a global set of used diffs.
        # However, removing/adding diffs for one row is O(k^2). 
        # Let's maintain `global_used` as a set.
        
        # For simplicity and speed, let's use a "repair-first" approach or just brute force check if n_rows*k is small.
        # n_rows=4, k=7 -> 28 vars. k^2 = 49 diffs per row. Total ~200 diffs.
        # Checking 200 diffs is very fast.
        
        # Let's implement a helper to check if a candidate row is valid given other rows.
        # Actually, it's easier to just compute the set of diffs for the modified row and check against the union of diffs of other rows.
        
        # Pre-compute diffs for other rows? That's expensive if done every step.
        # Let's maintain a global set `all_diffs` and a count or just re-compute if needed.
        # Given small size, recomputing the global set is O(N*K^2) ~ 4*49 = 200 ops.
        # This is very fast. Let's do it.
        
        # 1. Modify copy
        old_val = current_rows[r_idx][p_idx]
        current_rows[r_idx][p_idx] = new_val
        
        # Check row validity
        row = current_rows[r_idx]
        valid_row = True
        if row[p_idx - 1] >= new_val or (p_idx < k and row[p_idx + 1] <= new_val):
            valid_row = False
        
        # Check internal diffs of this row
        if valid_row:
            row_diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in row_diffs:
                        valid_row = False
                        break
                    row_diffs.add(d)
            if not valid_row:
                pass
            else:
                # Check against other rows
                # Compute all diffs from other rows
                other_diffs = set()
                for r in range(n_rows):
                    if r == r_idx:
                        continue
                    r_row = current_rows[r]
                    for i in range(k + 1):
                        for j in range(i + 1, k + 1):
                            other_diffs.add(r_row[j] - r_row[i])
                
                if row_diffs & other_diffs:
                    valid_row = False
        
        new_score = max(max(r) for r in current_rows)
        
        if valid_row:
            # Accept if better or with prob exp(-delta/T)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [r[:] for r in current_rows]
            else:
                # new_score >= current_score
                # Since we only change one value, score change is small.
                # If we increased a value, score might go up.
                # If we decreased, score might go down (handled above) or stay same.
                # If score is same, we accept with some prob to allow exploration?
                # Usually for minimization with hard constraints, we only accept if valid.
                # But to escape local minima, we might accept worse valid states.
                delta = new_score - current_score
                if delta > 0:
                    prob = pow(2.71828, -delta / T)
                    if random.random() < prob:
                        current_score = new_score
                else:
                    # delta == 0 (score unchanged but structure changed)
                    # Accept to explore
                    current_score = new_score
        else:
            # Invalid move, revert
            current_rows[r_idx][p_idx] = old_val
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart temperature if stuck?
            # Simple restart: if no improvement in last X steps, reset T
            # For now, just keep going.
            
    return best_rows
# EVOLVE-BLOCK-END
