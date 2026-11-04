import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_diffs(row, idx):
        v = row[idx]
        diffs = set()
        for i in range(idx):
            diffs.add(v - row[i])
        for i in range(idx + 1, len(row)):
            diffs.add(row[i] - v)
        return diffs

    def is_valid_change(row, idx, new_val, used, old_val):
        if idx > 0 and new_val <= row[idx - 1]:
            return False
        if idx < len(row) - 1 and new_val >= row[idx + 1]:
            return False
        
        old_diffs = get_diffs(row, idx)
        
        diff_list = []
        for i in range(idx):
            diff_list.append(new_val - row[i])
        for i in range(idx + 1, len(row)):
            diff_list.append(row[i] - new_val)
            
        if len(set(diff_list)) != len(diff_list):
            return False
            
        new_diffs = set(diff_list)
        if new_diffs & (used - old_diffs):
            return False
        return True

    def apply_change(row, idx, new_val, used, old_val):
        old_diffs = get_diffs(row, idx)
        used -= old_diffs
        row[idx] = new_val
        new_diffs = get_diffs(row, idx)
        used |= new_diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used and len(new_diffs) == len(row):
                        row.append(x)
                        used |= new_diffs
                        current_max = x
                        found = True
                        break
                    x += 1
                if not found:
                    break
            rows.append(row)
        return rows, used

    best_rows, best_used = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    rows = [r[:] for r in best_rows]
    used = set(best_used)
    
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        order = list(range(n_rows))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline: break
            row = rows[i]
            for j in range(k, 0, -1):
                if time.time() > deadline: break
                old_val = row[j]
                min_val = row[j-1] + 1
                # Try to decrease
                # Limit search depth to avoid long loops if stuck
                for new_val in range(old_val - 1, max(min_val - 1, old_val - 20), -1):
                    if is_valid_change(row, j, new_val, used, old_val):
                        apply_change(row, j, new_val, used, old_val)
                        improved = True
                        break
        
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
            best_used = set(used)
            
        if not improved:
            # Perturb
            if n_rows > 0 and k > 0:
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
                row = rows[r_idx]
                old_val = row[j_idx]
                for delta in range(1, 5):
                    new_val = old_val + delta
                    if j_idx < k and new_val >= row[j_idx + 1]:
                        continue
                    if is_valid_change(row, j_idx, new_val, used, old_val):
                        apply_change(row, j_idx, new_val, used, old_val)
                        break

    return best_rows
