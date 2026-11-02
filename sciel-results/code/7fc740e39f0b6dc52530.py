import time
import random
import math

def construct(n_rows, k):
    # 1. Greedy Initialization
    used_diffs = {}
    rows = []
    
    # We need a function to build one row given used_diffs
    def build_row(used_diffs, k, max_bound):
        row = [0]
        for _ in range(k):
            last = row[-1]
            placed = False
            for v in range(last + 1, max_bound + 1):
                valid = True
                new_diffs = []
                for a in row:
                    d = v - a
                    if used_diffs.get(d, 0) > 0:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                row.append(v)
                for d in new_diffs:
                    used_diffs[d] = used_diffs.get(d, 0) + 1
                placed = True
                break
            if not placed:
                return None
        return row

    # Try to initialize with a reasonable bound
    # Lower bound calculation
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
        
    init_bound = M_lower + 50
    
    while True:
        used_diffs = {}
        rows = []
        success = True
        for _ in range(n_rows):
            row = build_row(used_diffs, k, init_bound)
            if row is None:
                success = False
                break
            rows.append(row)
        if success:
            break
        init_bound += 20

    # 2. Simulated Annealing
    def compute_cost(rows):
        used = set()
        collisions = 0
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        collisions += 1
                    else:
                        used.add(d)
        return max_val + 100 * collisions

    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    T = 100.0
    T_min = 0.1
    cooling = 0.999
    deadline = time.time() + 110.0
    
    best_rows = [r[:] for r in rows]
    best_cost = compute_cost(best_rows)
    
    current_rows = [r[:] for r in rows]
    current_cost = best_cost
    
    while time.time() < deadline and T > T_min:
        # Identify max entry row
        max_idx = -1
        max_val = -1
        for i, r in enumerate(current_rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_idx = i
                
        # Bias: 50% chance to target the max row, 50% random row
        if random.random() < 0.5 and max_idx != -1:
            row_idx = max_idx
        else:
            row_idx = random.randint(0, n_rows - 1)
            
        # Pick position in row (1 to k)
        pos = random.randint(1, k)
        
        old_val = current_rows[row_idx][pos]
        
        # Determine valid range for new value
        min_val = current_rows[row_idx][pos-1] + 1
        # Allow moving up to current max + 10 to explore, but prefer smaller
        # If pos < k, must be < next element
        if pos < k:
            max_val_allowed = current_rows[row_idx][pos+1] - 1
        else:
            max_val_allowed = max_val + 10 # Allow increasing the max slightly to escape?
            
        if max_val_allowed < min_val:
            T *= cooling
            continue
            
        # Random new value
        # Bias towards smaller values to reduce max
        # Pick from [min_val, max_val_allowed]
        # To reduce max, we often want to decrease old_val.
        # So pick from [min_val, min(old_val, max_val_allowed)] with higher probability?
        # Or just uniform.
        new_val = random.randint(min_val, max_val_allowed)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        # Apply move
        current_rows[row_idx][pos] = new_val
        
        # Calculate new cost
        new_cost = compute_cost(current_rows)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[row_idx][pos] = old_val
            
        T *= cooling
        
    if not is_valid(best_rows):
        return rows
    return best_rows
