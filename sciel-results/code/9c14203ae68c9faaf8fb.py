import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(k, used_diffs, max_limit=None):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                if max_limit is not None and x > max_limit:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
        return row

    def random_greedy_set(n_rows, k, randomize_prob=0.4, randomize_width=5):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            success = True
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                while len(candidates) < randomize_width:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                    if x > 1000: 
                        success = False
                        break
                if not success:
                    return None
                
                if random.random() < randomize_prob and len(candidates) > 1:
                    chosen = random.choice(candidates)
                else:
                    chosen = candidates[0]
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    def local_improve(rows, n_rows, k, deadline):
        current_rows = [r[:] for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(n_rows))
            random.shuffle(indices)
            for i in indices:
                if time.time() >= deadline:
                    break
                others = [current_rows[j] for j in range(n_rows) if j != i]
                used_by_others = set()
                for r in others:
                    used_by_others |= get_diffs(r)
                
                new_row = greedy_row(k, used_by_others, max_limit=current_max)
                if new_row is not None:
                    new_row_max = max(new_row)
                    if new_row_max < max(current_rows[i]):
                        current_rows[i] = new_row
                        new_global_max = max(max(r) for r in current_rows)
                        if new_global_max < current_max:
                            current_max = new_global_max
                            improved = True
        return current_rows, current_max

    best_rows = None
    best_max_val = float('inf')
    
    # Initialize with a deterministic greedy solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows.append(row)
    best_rows = rows
    best_max_val = max(max(r) for r in best_rows)

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # 1. Try to improve the current best solution periodically
        if attempts % 5 == 0:
            improved_rows, improved_max = local_improve(best_rows, n_rows, k, deadline)
            if improved_max < best_max_val:
                best_max_val = improved_max
                best_rows = improved_rows
                if best_max_val <= 105:
                    break

        # 2. Generate a new random solution
        rows = random_greedy_set(n_rows, k)
        if rows is None:
            continue
        
        all_diffs = set()
        valid = True
        for r in rows:
            diffs = get_diffs(r)
            if all_diffs & diffs:
                valid = False
                break
            all_diffs |= diffs
        if not valid:
            continue
            
        current_max = max(max(r) for r in rows)
        
        if current_max < best_max_val or attempts < 10:
             improved_rows, improved_max = local_improve(rows, n_rows, k, deadline)
             if improved_max < best_max_val:
                 best_max_val = improved_max
                 best_rows = improved_rows
                 if best_max_val <= 105:
                     break

    return best_rows
