import time
import random
import sys

def construct(n_rows, k):
    # n_rows = 7, k = 5 (implied by "6 increasing integers")
    # Each row has k+1 elements: a[0]=0, a[1]...a[k]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate a baseline greedy solution
    # This ensures we always have a valid result to return
    used = set()
    baseline_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        baseline_rows.append(row)
    
    best_valid_rows = baseline_rows
    best_max_val = max(max(r) for r in baseline_rows)
    
    # 2. Simulated Annealing / Local Search
    # State: list of rows
    # We will try to improve on the baseline
    
    # Helper to compute max value
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    # Helper to check validity and count conflicts
    # Returns (is_valid, num_conflicts)
    def check_conflicts(rows):
        used = set()
        conflicts = 0
        for row in rows:
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    elif d in row_diffs:
                        # Duplicate within same row, also a conflict globally effectively
                        conflicts += 1
                    else:
                        used.add(d)
                        row_diffs.add(d)
        return conflicts == 0, conflicts

    # Initial state for SA: use baseline
    current_rows = [r[:] for r in baseline_rows]
    current_max = best_max_val
    current_conflicts = 0 # Baseline is valid
    
    # SA Parameters
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    
    # Precompute indices for moves
    # A move is (row_idx, pos_idx, new_value)
    # pos_idx is 1..k
    
    while time.time() < deadline:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][p_idx]
        
        # Propose new value
        # Bias towards lower values to reduce max? Or random?
        # Let's try random in a window
        # To avoid breaking strict increase, we need to check neighbors
        lower_bound = current_rows[r_idx][p_idx - 1] + 1
        upper_bound = current_rows[r_idx][p_idx + 1] - 1 if p_idx < k else old_val + 10
        
        # If old_val is the max, we might want to try lower values
        # If old_val is small, we might want to try higher values to resolve conflicts?
        # Actually, we want to minimize max.
        
        # Strategy: 
        # If we are valid, try to reduce max.
        # If we are invalid, try to reduce conflicts.
        
        # For simplicity, let's just do random perturbations and accept if better cost.
        # Cost = conflicts * 1000 + max_val
        
        # Generate candidate
        # Random step size
        step = random.randint(-5, 5)
        new_val = old_val + step
        
        # Enforce bounds
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            continue
            
        # Apply move temporarily
        current_rows[r_idx][p_idx] = new_val
        
        # Check validity
        # Efficient check: only recompute diffs for row r_idx and check against global used?
        # This is tricky with incremental updates.
        # Given small size (7*15=105 diffs), full check is fast enough in Python if optimized.
        # Let's try a fast full check.
        
        is_valid, confs = check_conflicts(current_rows)
        new_max = get_max_val(current_rows)
        
        # Calculate costs
        old_cost = current_conflicts * 1000 + current_max
        new_cost = confs * 1000 + new_max
        
        # Acceptance probability
        if new_cost < old_cost:
            accept = True
        else:
            delta = new_cost - old_cost
            if T > 0:
                # Accept with probability exp(-delta / T)
                if random.random() < pow(2.71828, -delta / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            current_conflicts = confs
            current_max = new_max
            # Update best valid
            if is_valid:
                if new_max < best_max_val:
                    best_max_val = new_max
                    best_valid_rows = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[r_idx][p_idx] = old_val
            
        # Cool down
        T *= alpha
        
        # If we found a solution with max 105 (the bound), we can stop
        if best_max_val == 105:
            break
            
    return best_valid_rows
