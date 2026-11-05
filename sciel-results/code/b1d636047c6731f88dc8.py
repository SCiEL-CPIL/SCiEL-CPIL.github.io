import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # 1. Greedy Initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check differences with existing elements in row
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    curr_rows = greedy_init()
    
    # 2. Compute Initial State
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    curr_counts = get_diff_counts(curr_rows)
    curr_penalty = get_penalty(curr_counts)
    curr_M = get_M(curr_rows)
    
    best_rows = [r[:] for r in curr_rows]
    best_M = curr_M
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    lambda_pen = 100.0 # High penalty to ensure validity
    
    curr_energy = curr_M + lambda_pen * curr_penalty
    
    # 3. SA Loop
    i = 0
    while time.time() < deadline:
        i += 1
        
        # Generate Move
        r_idx = random.randint(0, n_rows - 1)
        row = curr_rows[r_idx]
        
        # Pick an index to change. Index 0 is fixed at 0.
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        
        # Determine valid range for new value
        low = row[j_idx - 1] + 1
        if j_idx < k:
            high = row[j_idx + 1] - 1
        else:
            high = old_val # Bias towards shrinking the last element
        
        if high < low:
            continue
            
        # Random step biased towards small changes
        step = random.randint(-3, 3)
        new_val = old_val + step
        
        # Clamp to range
        if new_val < low: new_val = low
        if new_val > high: new_val = high
        
        if new_val == old_val:
             # If clamped to same value, try a different random value in range
             if high > low:
                 new_val = random.randint(low, high)
             else:
                 continue
        
        # Calculate change in differences
        old_diffs = set()
        for i in range(j_idx):
            d = old_val - row[i]
            old_diffs.add(d)
            curr_counts[d] -= 1
            if curr_counts[d] == 0:
                del curr_counts[d]
                
        for h in range(j_idx + 1, k + 1):
            d = row[h] - old_val
            old_diffs.add(d)
            curr_counts[d] -= 1
            if curr_counts[d] == 0:
                del curr_counts[d]
        
        # Tentatively set new value
        row[j_idx] = new_val
        
        new_diffs = set()
        for i in range(j_idx):
            d = new_val - row[i]
            new_diffs.add(d)
            curr_counts[d] = curr_counts.get(d, 0) + 1
            
        for h in range(j_idx + 1, k + 1):
            d = row[h] - new_val
            new_diffs.add(d)
            curr_counts[d] = curr_counts.get(d, 0) + 1
            
        new_penalty = get_penalty(curr_counts)
        new_M = max(max(r) for r in curr_rows)
        
        new_energy = new_M + lambda_pen * new_penalty
        
        delta_E = new_energy - curr_energy
        
        # Acceptance criterion
        if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
            curr_energy = new_energy
            curr_penalty = new_penalty
            curr_M = new_M
            
            if curr_penalty == 0:
                if curr_M < best_M:
                    best_M = curr_M
                    best_rows = [r[:] for r in curr_rows]
        else:
            # Revert
            row[j_idx] = old_val
            
            # Restore counts
            for d in new_diffs:
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            for d in old_diffs:
                curr_counts[d] = curr_counts.get(d, 0) + 1

        # Cool down
        if i % 100 == 0:
            T *= cooling
            if T < 0.1:
                T = 0.1

    return best_rows
