import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Initialize with Greedy
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
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    best_scope = get_scope(best_rows)
    
    # Initial diff count
    def calc_diff_counts(rows):
        diff_count = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        return diff_count

    current_diff_count = calc_diff_counts(current_rows)
    
    def count_conflicts(diff_count):
        return sum(c - 1 for c in diff_count.values() if c > 1)
        
    current_conflicts = count_conflicts(current_diff_count)
    current_scope = get_scope(current_rows)
    penalty = 100
    current_cost = current_scope + penalty * current_conflicts
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        min_val = row[j_idx - 1] + 1
        if j_idx < k:
            max_val = row[j_idx + 1] - 1
        else:
            max_val = best_scope + 20 # Allow some expansion
            
        if min_val > max_val:
            continue
            
        new_val = random.randint(min_val, max_val)
        if new_val == old_val:
            continue
            
        # Update diff_count incrementally
        # Remove old diffs
        for i in range(j_idx):
            d = old_val - row[i]
            current_diff_count[d] -= 1
            if current_diff_count[d] == 0:
                del current_diff_count[d]
                
        for i in range(j_idx + 1, len(row)):
            d = row[i] - old_val
            current_diff_count[d] -= 1
            if current_diff_count[d] == 0:
                del current_diff_count[d]
                
        # Apply change
        row[j_idx] = new_val
        
        # Add new diffs
        for i in range(j_idx):
            d = new_val - row[i]
            current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
        for i in range(j_idx + 1, len(row)):
            d = row[i] - new_val
            current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
        new_conflicts = count_conflicts(current_diff_count)
        new_scope = get_scope(current_rows)
        new_cost = new_scope + penalty * new_conflicts
        
        delta_cost = new_cost - current_cost
        
        if delta_cost <= 0 or random.random() < pow(2.71828, -delta_cost / T):
            current_cost = new_cost
            current_conflicts = new_conflicts
            current_scope = new_scope
            
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            row[j_idx] = old_val
            # Revert diff_count
            for i in range(j_idx):
                d = new_val - row[i]
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
                    
            for i in range(j_idx + 1, len(row)):
                d = row[i] - new_val
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
                    
            for i in range(j_idx):
                d = old_val - row[i]
                current_diff_count[d] = current_diff_count.get(d, 0) + 1
                
            for i in range(j_idx + 1, len(row)):
                d = row[i] - old_val
                current_diff_count[d] = current_diff_count.get(d, 0) + 1

        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_rows
