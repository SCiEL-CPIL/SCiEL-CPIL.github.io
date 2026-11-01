import random
import time
import math

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    timeout = 110
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def get_all_diffs(rows, exclude_row=-1):
        used = set()
        for i, row in enumerate(rows):
            if i == exclude_row:
                continue
            used.update(get_diffs(row))
        return used

    def greedy_construct(random_seed=0):
        random.seed(random_seed)
        rows = [[0] for _ in range(n_rows)]
        used = set()
        
        for _ in range(n_rows * k):
            candidates = []
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    x = rows[i][-1] + 1
                    limit = 10000 # Safety limit
                    while x <= limit:
                        new_diffs = {x - v for v in rows[i]}
                        if not (new_diffs & used):
                            candidates.append((x, i, new_diffs))
                            break
                        x += 1
            
            if not candidates:
                return None
            
            # Randomized choice among top candidates
            # Sort by value
            candidates.sort(key=lambda c: c[0])
            # Pick the best, or sometimes a close one
            if len(candidates) > 1 and random.random() < 0.1:
                # Pick from top 3
                top_k = min(3, len(candidates))
                best_val, best_idx, new_diffs = candidates[random.randint(0, top_k - 1)]
            else:
                best_val, best_idx, new_diffs = candidates[0]
                
            rows[best_idx].append(best_val)
            used.update(new_diffs)
            
        return rows

    def local_search(rows):
        current_rows = [r[:] for r in rows]
        
        while time.time() - start_time < timeout:
            improved = False
            indices = list(range(n_rows))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > timeout:
                    break
                other_diffs = get_all_diffs(current_rows, exclude_row=i)
                
                for j in range(k, 0, -1):
                    old_val = current_rows[i][j]
                    new_val = old_val - 1
                    while new_val > current_rows[i][j-1]:
                        candidate_row = current_rows[i][:j] + [new_val] + current_rows[i][j+1:]
                        new_diffs = get_diffs(candidate_row)
                        
                        if new_diffs.isdisjoint(other_diffs):
                            current_rows[i] = candidate_row
                            improved = True
                            break
                        new_val -= 1
                    if improved:
                        break
            if not improved:
                break
                
        return current_rows

    # Initial deterministic greedy
    best_solution = None
    # Try a few seeds for initial construction
    for s in range(10):
        sol = greedy_construct(s)
        if sol:
            if best_solution is None:
                best_solution = sol
            else:
                if max(max(r) for r in sol) < max(max(r) for r in best_solution):
                    best_solution = sol
            if time.time() - start_time > timeout / 2:
                break

    if best_solution is None:
        # Fallback
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
            i = target_row_idx
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - v for v in rows[i]}
                if not (new_diffs & used):
                    rows[i].append(x)
                    used.update(new_diffs)
                    break
                x += 1
        best_solution = rows

    best_cost = max(max(r) for r in best_solution)
    
    # Local Search Phase
    seed = 100
    while time.time() - start_time < timeout:
        candidate = greedy_construct(seed)
        if candidate is not None:
            candidate = local_search(candidate)
            cost = max(max(r) for r in candidate)
            if cost < best_cost:
                best_cost = cost
                best_solution = candidate
        seed += 1
        
    return best_solution
