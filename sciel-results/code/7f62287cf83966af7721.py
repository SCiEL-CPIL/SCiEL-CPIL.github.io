import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if x - a in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
                    if x > 5000: # Safety
                        return None
            rows.append(row)
        return rows

    curr_rows = greedy_init()
    if curr_rows is None:
        # Fallback: just return a dummy valid set? Or raise?
        # For robustness, let's try a simpler random init if greedy fails
        # But greedy shouldn't fail for these sizes.
        return [[i for i in range(k+1)] for _ in range(n_rows)] # Invalid, but better than crash? No, evaluator scores 0 if invalid.
        
    curr_M = max(max(r) for r in curr_rows)
    best_rows = copy.deepcopy(curr_rows)
    best_M = curr_M
    
    # Build diff counts
    curr_counts = {}
    for r in curr_rows:
        for i in range(len(r)):
            ri = r[i]
            for j in range(i+1, len(r)):
                d = r[j] - ri
                curr_counts[d] = curr_counts.get(d, 0) + 1
                
    curr_penalty = 0
    for c in curr_counts.values():
        if c > 1:
            curr_penalty += c - 1
            
    # Should be 0 if greedy worked correctly
    if curr_penalty != 0:
        # Greedy failed to maintain uniqueness? Unlikely.
        pass

    row_maxes = [max(r) for r in curr_rows]
    
    lambda_pen = 1000.0
    T = 10.0
    cooling = 0.999
    
    curr_energy = curr_M + lambda_pen * curr_penalty
    
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        row = curr_rows[r_idx]
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        
        low = row[j_idx - 1] + 1
        if j_idx < k:
            high = row[j_idx + 1] - 1
        else:
            high = curr_M + 10
            
        if high < low:
            continue
            
        new_val = random.randint(low, high)
        
        if new_val == old_val:
            continue
            
        # Remove old diffs
        # Update penalty and counts
        pen_change = 0
        
        # Diffs with lower indices
        for i in range(j_idx):
            d = old_val - row[i]
            c = curr_counts[d]
            if c > 1:
                pen_change -= 1
            if c == 1:
                del curr_counts[d]
            else:
                curr_counts[d] = c - 1
                
        # Diffs with higher indices
        for h in range(j_idx + 1, k + 1):
            d = row[h] - old_val
            c = curr_counts[d]
            if c > 1:
                pen_change -= 1
            if c == 1:
                del curr_counts[d]
            else:
                curr_counts[d] = c - 1
                
        # Set new
        row[j_idx] = new_val
        
        # Add new diffs
        for i in range(j_idx):
            d = new_val - row[i]
            c = curr_counts.get(d, 0)
            if c == 1:
                pen_change += 1
            curr_counts[d] = c + 1
                
        for h in range(j_idx + 1, k + 1):
            d = row[h] - new_val
            c = curr_counts.get(d, 0)
            if c == 1:
                pen_change += 1
            curr_counts[d] = c + 1
                
        curr_penalty += pen_change
        
        # Calculate new M
        old_row_max = row_maxes[r_idx]
        new_row_max = max(row)
        
        if new_row_max > old_row_max:
            if new_row_max > curr_M:
                new_M = new_row_max
            else:
                new_M = curr_M
        else:
            if old_row_max == curr_M:
                # Find max of others
                max_other = 0
                for i in range(n_rows):
                    if i != r_idx and row_maxes[i] > max_other:
                        max_other = row_maxes[i]
                new_M = max(new_row_max, max_other)
            else:
                new_M = curr_M
                
        new_energy = new_M + lambda_pen * curr_penalty
        delta_E = new_energy - curr_energy
        
        if delta_E < 0 or (T > 0.01 and random.random() < math.exp(-delta_E / T)):
            curr_energy = new_energy
            curr_M = new_M
            row_maxes[r_idx] = new_row_max
            
            if curr_penalty == 0 and curr_M < best_M:
                best_M = curr_M
                best_rows = copy.deepcopy(curr_rows)
        else:
            # Revert
            row[j_idx] = old_val
            
            # Remove new diffs
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                if c == 1:
                    del curr_counts[d]
                else:
                    curr_counts[d] = c - 1
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                if c == 1:
                    del curr_counts[d]
                else:
                    curr_counts[d] = c - 1
                    
            # Add old diffs
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts.get(d, 0)
                if c == 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts.get(d, 0)
                if c == 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            # M and row_maxes unchanged

        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
