import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # 1. Greedy Initialization
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

    current_rows = greedy_init()
    best_rows = [r[:] for r in current_rows]
    
    # Helper to calculate cost: conflicts * 10000 + max_value
    def get_cost(rows):
        all_diffs = []
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        counts = Counter(all_diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += 1
        
        return conflicts * 10000 + max_val, conflicts, max_val

    # Helper to check validity quickly (early exit)
    def is_valid(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    curr_cost, curr_conf, curr_max = get_cost(current_rows)
    best_cost, best_conf, best_max = curr_cost, curr_conf, curr_max

    # 2. Simulated Annealing
    T = 500.0
    T_min = 1.0
    alpha = 0.995  # Cooling factor
    
    # Pre-calculate bounds for random moves
    # We allow expansion up to best_max + 50 to escape local minima
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Propose move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)  # Index 0 is fixed at 0
        
        row = current_rows[r_idx]
        old_val = row[i_idx]
        
        # Determine range for new value
        lower = row[i_idx - 1] + 1
        # Upper bound: allow some expansion beyond current best to explore
        upper_bound = max(best_max, curr_max) + 50
        
        if lower > upper_bound:
            continue
            
        new_val = random.randint(lower, upper_bound)
        if new_val == old_val:
            continue
            
        # Apply move
        row[i_idx] = new_val
        
        new_cost, new_conf, new_max = get_cost(current_rows)
        
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            curr_cost = new_cost
            curr_conf = new_conf
            curr_max = new_max
            
            if curr_conf == 0 and curr_cost < best_cost:
                best_cost = curr_cost
                best_conf = curr_conf
                best_max = curr_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject
            row[i_idx] = old_val
            
        T *= alpha

    # 3. Final Polish
    # Deterministic local search to reduce scope
    def polish(rows):
        rows = [r[:] for r in rows]
        improved = True
        # Limit passes to avoid infinite loops if time runs out
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce from right to left
                for i in range(k, 0, -1):
                    old_val = row[i]
                    min_valid = old_val
                    # Try values from row[i-1]+1 to old_val-1
                    for v in range(row[i-1] + 1, old_val):
                        row[i] = v
                        if is_valid(rows):
                            min_valid = v
                            break
                        # If invalid, the loop continues to the next (larger) v
                    row[i] = min_valid
                    if min_valid < old_val:
                        improved = True
        return rows

    if time.time() < deadline - 1.0:
        polished_rows = polish(best_rows)
        # Verify the polished rows are valid
        if is_valid(polished_rows):
            best_rows = polished_rows
            # Update best_cost just in case, though we only return rows
            # The evaluator calculates the score from the rows

    return best_rows
