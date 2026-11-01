import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    def is_valid_row(row):
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        return True

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct(n_rows, k, randomize=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows, used

    def local_search(rows, max_time=30):
        current_rows = [r[:] for r in rows]
        used = set()
        for row in current_rows:
            used |= get_row_diffs(row)
        
        current_cost = max(max(r) for r in current_rows)
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9999
        start = time.time()
        
        while time.time() - start < max_time and T > T_min:
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, row_len - 1)
            
            row = current_rows[r_idx]
            low = row[p_idx - 1] + 1
            high = row[p_idx + 1] - 1 if p_idx < row_len - 1 else current_cost + 20
            
            if high < low:
                T *= cooling
                continue
                
            new_val = random.randint(low, high)
            if new_val == row[p_idx]:
                T *= cooling
                continue
            
            # Remove old diffs
            old_val = row[p_idx]
            old_diffs = set()
            for j, v in enumerate(row):
                if j != p_idx:
                    old_diffs.add(abs(old_val - v))
            
            # Check new diffs
            new_diffs = set()
            for j, v in enumerate(row):
                if j != p_idx:
                    new_diffs.add(abs(new_val - v))
            
            # Check validity
            valid = True
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            # Apply move
            row[p_idx] = new_val
            used -= old_diffs
            used |= new_diffs
            
            new_cost = max(max(r) for r in current_rows)
            delta = new_cost - current_cost
            
            if delta <= 0:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [r[:] for r in current_rows]
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    row[p_idx] = old_val
                    used -= new_diffs
                    used |= old_diffs
            
            T *= cooling
        
        return best_rows, best_cost

    start_global = time.time()
    max_total_time = 110
    
    # First, get a baseline solution
    base_rows, _ = greedy_construct(n_rows, k)
    best_rows, best_cost = local_search(base_rows, max_time=30)
    
    # Now try to improve with different strategies
    for restart in range(5):
        if time.time() - start_global > max_total_time - 30:
            break
        
        # Generate a new random-ish starting point
        rows, _ = greedy_construct(n_rows, k)
        # Perturb the rows
        for r in rows:
            for i in range(1, len(r)):
                if random.random() < 0.3:
                    r[i] += random.randint(0, 10)
        
        # Run local search
        new_rows, new_cost = local_search(rows, max_time=20)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_rows = new_rows
    
    return best_rows
