"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    
    # Helper to compute difference bitmask for a row
    def row_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # 1. Greedy Baseline to get a valid starting point
    def greedy_solution():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new |= (1 << d)
                if not conflict:
                    row_mask_current = row_mask(row)
                    if new & row_mask_current:
                        continue
                    if new & used:
                        continue
                    row.append(x)
            used |= row_mask(row)
            rows.append(row)
        return rows

    # Initial Solution
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # Current State for SA
    current_rows = [list(r) for r in best_rows]
    current_max = best_max
    
    # Precompute masks for current rows
    row_masks = [row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m

    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    def propose_neighbor(rows, row_masks, global_mask):
        """
        Returns (new_rows, new_row_masks, new_global_mask, new_max) or None
        """
        r_idx = random.randint(0, n_rows - 1)
        row = list(rows[r_idx])
        
        if random.random() < 0.7:
            # Shrink: Reduce a random element (preferably large ones)
            if random.random() < 0.5:
                pos = k
                old_val = row[pos]
                min_val = row[pos-1] + 1
                if old_val <= min_val:
                    return None
                offset = random.randint(1, min(10, old_val - min_val))
                new_val = old_val - offset
            else:
                pos = random.randint(1, k-1)
                old_val = row[pos]
                min_val = row[pos-1] + 1
                max_val = row[pos+1] - 1
                if max_val < min_val:
                    return None
                new_val = random.randint(min_val, max_val)
            
            row[pos] = new_val
            
            new_mask = row_mask(row)
            if new_mask.bit_count() != diffs_per_row:
                return None
            
            other_mask = global_mask & ~row_masks[r_idx]
            if new_mask & other_mask:
                return None
            
            new_rows = [list(r) if i != r_idx else row for i, r in enumerate(rows)]
            new_row_masks = list(row_masks)
            new_row_masks[r_idx] = new_mask
            new_global_mask = (global_mask & ~row_masks[r_idx]) | new_mask
            new_max = max(max(r) for r in new_rows)
            
            return (new_rows, new_row_masks, new_global_mask, new_max)
            
        else:
            # Perturb: Change a random element to a random value
            pos = random.randint(1, k)
            old_val = row[pos]
            
            if pos == 1:
                min_val = 1
            else:
                min_val = row[pos-1] + 1
                
            if pos == k:
                max_val = old_val + 10
            else:
                max_val = row[pos+1] - 1
                
            if max_val < min_val:
                return None
            
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                return None
                
            row[pos] = new_val
            
            new_mask = row_mask(row)
            if new_mask.bit_count() != diffs_per_row:
                return None
            
            other_mask = global_mask & ~row_masks[r_idx]
            if new_mask & other_mask:
                return None
            
            new_rows = [list(r) if i != r_idx else row for i, r in enumerate(rows)]
            new_row_masks = list(row_masks)
            new_row_masks[r_idx] = new_mask
            new_global_mask = (global_mask & ~row_masks[r_idx]) | new_mask
            new_max = max(max(r) for r in new_rows)
            
            return (new_rows, new_row_masks, new_global_mask, new_max)

    # SA Loop
    while time.time() < deadline and T > min_T:
        neighbor = propose_neighbor(current_rows, row_masks, global_mask)
        if neighbor is None:
            continue
            
        n_rows_list, n_masks, n_global, n_max = neighbor
        delta = n_max - current_max
        
        if delta <= 0:
            current_rows = n_rows_list
            row_masks = n_masks
            global_mask = n_global
            current_max = n_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
        else:
            if random.random() < math.exp(-delta / T):
                current_rows = n_rows_list
                row_masks = n_masks
                global_mask = n_global
                current_max = n_max
                
        T *= cooling
        
    # Final validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best_rows):
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)

    return best_rows

# EVOLVE-BLOCK-END
