import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def get_row_diffs_bitset(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs
    
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            if randomize:
                if len(row) > 2 and random.random() < 0.3:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    def find_solution_backtrack(M, deadline_t, randomize_order=False):
        def solve_row(idx, global_used):
            if idx == n_rows:
                return []
            
            def build(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    if row_used & global_used:
                        return None
                    res = solve_row(idx + 1, global_used | row_used)
                    if res is None:
                        return None
                    return [vals] + res
                
                last = vals[-1]
                candidates = list(range(last + 1, M + 1))
                if randomize_order:
                    random.shuffle(candidates)
                
                for v in candidates:
                    new_diffs = 0
                    conflict = False
                    for u in vals:
                        d = v - u
                        if (row_used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if conflict:
                        continue
                    
                    if new_diffs & global_used:
                        continue
                    
                    vals.append(v)
                    res = build(pos + 1, vals, row_used | new_diffs)
                    if res is not None:
                        return res
                    vals.pop()
                
                return None
            
            return build(1, [0], 0)
        
        return solve_row(0, 0)
    
    def local_search(rows, deadline_t, max_iter=100):
        rows = [list(r) for r in rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            row_idx = random.randint(0, n_rows - 1)
            row = list(rows[row_idx])
            
            current_max = row[-1]
            improved = False
            
            for new_max in range(max(2, current_max - 1), 0, -1):
                for attempt in range(10):
                    new_row = [0]
                    for pos in range(1, k + 1):
                        candidates = list(range(1, new_max + 1))
                        random.shuffle(candidates)
                        found = False
                        for v in candidates:
                            if v <= new_row[-1]:
                                continue
                            new_diffs = {v - a for a in new_row}
                            if new_diffs & used_diffs:
                                continue
                            if len(new_diffs) != len(new_row):
                                continue
                            new_row.append(v)
                            found = True
                            break
                        if not found:
                            break
                    
                    if len(new_row) == k + 1 and new_row[-1] <= new_max:
                        row_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                row_diffs.add(new_row[j] - new_row[i])
                        
                        old_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                old_diffs.add(row[j] - row[i])
                        
                        used_diffs -= old_diffs
                        
                        if not (row_diffs & used_diffs):
                            used_diffs |= row_diffs
                            rows[row_idx] = new_row
                            new_max_entry = get_max_entry(rows)
                            if new_max_entry < best_max:
                                best_max = new_max_entry
                                best_rows = [list(r) for r in rows]
                            improved = True
                            break
                        
                        used_diffs -= row_diffs
                        used_diffs |= old_diffs
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                if n_rows > 1:
                    row_idx1, row_idx2 = random.sample(range(n_rows), 2)
                    row1 = list(rows[row_idx1])
                    row2 = list(rows[row_idx2])
                    
                    if len(row1) > 1 and len(row2) > 1:
                        i1 = random.randint(1, len(row1) - 1)
                        i2 = random.randint(1, len(row2) - 1)
                        
                        diffs1 = set()
                        for i in range(len(row1)):
                            for j in range(i+1, len(row1)):
                                diffs1.add(row1[j] - row1[i])
                        diffs2 = set()
                        for i in range(len(row2)):
                            for j in range(i+1, len(row2)):
                                diffs2.add(row2[j] - row2[i])
                        
                        used_diffs -= diffs1
                        used_diffs -= diffs2
                        
                        row1[i1], row2[i2] = row2[i2], row1[i1]
                        
                        valid = True
                        new_diffs1 = set()
                        for i in range(len(row1)):
                            for j in range(i+1, len(row1)):
                                d = row1[j] - row1[i]
                                if d in new_diffs1:
                                    valid = False
                                    break
                                new_diffs1.add(d)
                        if valid:
                            new_diffs2 = set()
                            for i in range(len(row2)):
                                for j in range(i+1, len(row2)):
                                    d = row2[j] - row2[i]
                                    if d in new_diffs2:
                                        valid = False
                                        break
                                    new_diffs2.add(d)
                        
                        if valid and not (new_diffs1 & new_diffs2) and not (new_diffs1 & used_diffs) and not (new_diffs2 & used_diffs):
                            if all(row1[i] < row1[i+1] for i in range(len(row1)-1)) and \
                               all(row2[i] < row2[i+1] for i in range(len(row2)-1)):
                                rows[row_idx1] = row1
                                rows[row_idx2] = row2
                                used_diffs |= new_diffs1
                                used_diffs |= new_diffs2
                                new_max_entry = get_max_entry(rows)
                                if new_max_entry < best_max:
                                    best_max = new_max_entry
                                    best_rows = [list(r) for r in rows]
                                improved = True
                        
                        if not improved:
                            used_diffs |= diffs1
                            used_diffs |= diffs2
                            rows[row_idx1] = list(row1)
                            rows[row_idx2] = list(row2)
        
        return best_rows, best_max
    
    search_deadline = deadline - 5.0
    
    # Phase 1: Try backtracking with various M values
    candidates = list(range(lower_bound, best_max))
    random.shuffle(candidates)
    
    for M in candidates[:min(20, len(candidates))]:
        if time.time() > search_deadline:
            break
        
        remaining_time = search_deadline - time.time()
        time_slice = min(remaining_time / 20, 5.0)
        time_slice = max(time_slice, 1.0)
        
        deadline_for_this = time.time() + time_slice
        
        try:
            sol = find_solution_backtrack(M, deadline_for_this, randomize_order=True)
            if sol:
                curr_max = get_max_entry(sol)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = sol
                    break
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 2: Local search to improve current best
    if time.time() < search_deadline:
        try:
            improved_rows, improved_max = local_search(best_rows, search_deadline, max_iter=100)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Phase 3: More randomized restarts if time allows
    if time.time() < deadline - 10:
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = restart * 1000 + 42
            greedy_rows = greedy_solution(randomize=True, seed=seed)
            greedy_max = get_max_entry(greedy_rows)
            
            if greedy_max < best_max:
                best_max = greedy_max
                best_rows = greedy_rows
            
            if time.time() < deadline - 5:
                try:
                    improved_rows, improved_max = local_search(greedy_rows, deadline - 5, max_iter=30)
                    if improved_max < best_max:
                        best_max = improved_max
                        best_rows = improved_rows
                except Exception:
                    pass
    
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
