import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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

    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)
    
    # State for SA
    rows = [r[:] for r in best_rows]
    
    # Precompute diff counts
    def get_diff_counts(rs):
        counts = {}
        for r in rs:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    diff_counts = get_diff_counts(rows)
    
    def get_collisions(counts):
        return sum(v - 1 for v in counts.values() if v > 1)
    
    current_collisions = get_collisions(diff_counts)
    current_scope = best_scope
    current_cost = current_scope + 1000 * current_collisions
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline and T > T_min:
        # Generate move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k) # 1 to k (0 is fixed)
        
        row = rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new value
        lower = row[j_idx - 1] + 1
        upper = row[j_idx + 1] - 1 if j_idx < k else old_val + 50
        
        if lower > upper:
            T *= alpha
            continue
            
        # Random new value
        # Use a random step size
        step = random.gauss(0, 10)
        new_val = int(old_val + step)
        
        # Clamp to valid range
        if new_val < lower:
            new_val = lower + random.randint(0, max(0, upper - lower))
        elif new_val > upper:
            new_val = upper - random.randint(0, max(0, upper - lower))
            
        if new_val == old_val:
            continue
            
        # Check if new_val is integer and valid
        # It is.
        
        # Calculate impact on diff_counts
        # Remove old diffs involving j_idx
        for i in range(k + 1):
            if i == j_idx:
                continue
            d = abs(row[j_idx] - row[i])
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Update row
        row[j_idx] = new_val
        
        # Add new diffs
        for i in range(k + 1):
            if i == j_idx:
                continue
            d = abs(new_val - row[i])
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_collisions = get_collisions(diff_counts)
        new_scope = max(max(r) for r in rows)
        new_cost = new_scope + 1000 * new_collisions
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_cost = new_cost
            current_collisions = new_collisions
            current_scope = new_scope
            
            if new_collisions == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            # Reject, revert
            # Remove new diffs
            for i in range(k + 1):
                if i == j_idx:
                    continue
                d = abs(new_val - row[i])
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            # Restore old val
            row[j_idx] = old_val
            
            # Add old diffs
            for i in range(k + 1):
                if i == j_idx:
                    continue
                d = abs(old_val - row[i])
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        T *= alpha
        
    return best_rows
