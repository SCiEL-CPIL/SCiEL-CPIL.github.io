import time

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task contract says 120s.
    # Use 110s to be safe.
    deadline = start_time + 110.0
    
    best_scope = [float('inf')]
    best_solution = [[]]
    
    # 1. Greedy Initial Solution
    used_greedy = set()
    rows_greedy = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_greedy:
                row.append(x)
                used_greedy |= new_diffs
        rows_greedy.append(row)
        
    if rows_greedy:
        best_scope[0] = max(max(r) for r in rows_greedy)
        best_solution[0] = [list(r) for r in rows_greedy]
        
    # 2. Backtracking Search
    rows_completed = []
    
    def search(row_idx, current_row, used_mask, prev_second):
        if time.time() > deadline:
            return
            
        if row_idx == n_rows:
            current_max = 0
            for r in rows_completed:
                if r:
                    m = r[-1]
                    if m > current_max:
                        current_max = m
            if current_max < best_scope[0]:
                best_scope[0] = current_max
                best_solution[0] = [list(r) for r in rows_completed]
            return
        
        if len(current_row) == 0:
            current_row.append(0)
            start_val = max(1, prev_second)
            max_val = best_scope[0] - (k - 1)
            
            for x in range(start_val, max_val + 1):
                d = x
                if d >= 2048:
                    continue
                if (used_mask >> d) & 1:
                    continue
                
                new_mask = used_mask | (1 << d)
                current_row.append(x)
                
                search(row_idx, current_row, new_mask, prev_second)
                
                current_row.pop()
            
            current_row.pop()
            return
        else:
            last = current_row[-1]
            start_val = last + 1
            remaining = k + 1 - len(current_row)
            max_val = best_scope[0] - (remaining - 1)
            
            for x in range(start_val, max_val + 1):
                conflict = False
                new_mask = used_mask
                for a in current_row:
                    d = x - a
                    if d >= 2048:
                        continue
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_mask |= (1 << d)
                
                if conflict:
                    continue
                
                current_row.append(x)
                
                if len(current_row) == k + 1:
                    rows_completed.append(list(current_row))
                    search(row_idx + 1, [], new_mask, current_row[1])
                    rows_completed.pop()
                else:
                    search(row_idx, current_row, new_mask, prev_second)
                
                current_row.pop()

    search(0, [], 0, 0)
    
    return best_solution[0]
