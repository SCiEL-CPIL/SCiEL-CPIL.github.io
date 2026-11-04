import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    conflict = False
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        return rows

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def score(rows):
        return max(max(r) for r in rows)

    best_rows = greedy_init()
    best_score = score(best_rows)
    
    def local_search(rows, max_iters):
        rows = [r[:] for r in rows]
        global_diffs = set()
        for r in rows:
            rd = get_diffs(r)
            global_diffs.update(rd)
        
        best_local = [r[:] for r in rows]
        best_local_score = score(rows)
        current_score = best_local_score
        
        T = 10.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
                
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            
            old_val = rows[r_idx][j_idx]
            
            low = rows[r_idx][j_idx-1] + 1
            high = rows[r_idx][j_idx+1] - 1 if j_idx < k else old_val + 5
            
            if low > high:
                continue
                
            # Remove old diffs
            old_diffs = set()
            for i in range(j_idx):
                old_diffs.add(old_val - rows[r_idx][i])
            for l in range(j_idx + 1, len(rows[r_idx])):
                old_diffs.add(rows[r_idx][l] - old_val)
            
            for d in old_diffs:
                global_diffs.discard(d)
            
            # Try to find a new valid value
            found_val = None
            new_diffs = set()
            
            # Try lower values first
            for new_val in range(low, min(old_val, high) + 1):
                valid = True
                new_diffs = set()
                for i in range(j_idx):
                    d = new_val - rows[r_idx][i]
                    if d in global_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    for l in range(j_idx + 1, len(rows[r_idx])):
                        d = rows[r_idx][l] - new_val
                        if d in global_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    found_val = new_val
                    break
            
            # If no lower value found, try a random higher value
            if found_val is None and high > old_val:
                candidates = range(old_val + 1, high + 1)
                if len(candidates) > 0:
                    new_val = random.choice(candidates)
                    valid = True
                    new_diffs = set()
                    for i in range(j_idx):
                        d = new_val - rows[r_idx][i]
                        if d in global_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        for l in range(j_idx + 1, len(rows[r_idx])):
                            d = rows[r_idx][l] - new_val
                            if d in global_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid:
                        found_val = new_val
            
            if found_val is not None:
                # Apply move
                rows[r_idx][j_idx] = found_val
                global_diffs.update(new_diffs)
                
                new_score = score(rows)
                
                accept = False
                if new_score < current_score:
                    accept = True
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = [r[:] for r in rows]
                elif new_score == current_score:
                    accept = True
                else:
                    delta = new_score - current_score
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accept = True
                        current_score = new_score
                        
                if not accept:
                    # Revert
                    global_diffs.difference_update(new_diffs)
                    global_diffs.update(old_diffs)
                    rows[r_idx][j_idx] = old_val
            else:
                # Revert
                global_diffs.update(old_diffs)
                
            T *= cooling
            
        return best_local

    # Run multiple rounds
    for restart in range(10):
        if time.time() - start_time > time_limit - 10:
            break
            
        random.seed(restart * 100)
        rows_to_opt = greedy_init()
        
        improved_rows = local_search(rows_to_opt, 5000)
        
        if score(improved_rows) < best_score:
            best_score = score(improved_rows)
            best_rows = improved_rows
            
    return best_rows
