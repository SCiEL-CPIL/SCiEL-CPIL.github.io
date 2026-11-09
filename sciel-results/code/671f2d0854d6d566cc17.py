import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Greedy initial solution
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Backtracking search with deadline guard
    def search():
        nonlocal best_rows, best_max, deadline
        if time.time() > deadline:
            return
        
        used = set()
        rows = []
        
        def backtrack(row_idx):
            nonlocal best_rows, best_max, deadline
            if time.time() > deadline:
                return
            
            if row_idx == n_rows:
                cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [r[:] for r in rows]
                return
            
            row = [0]
            
            def extend(pos):
                nonlocal used, best_rows, best_max, deadline
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    new_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in used:
                                return False
                            new_diffs.add(d)
                    
                    used |= new_diffs
                    rows.append(row[:])
                    backtrack(row_idx + 1)
                    rows.pop()
                    used -= new_diffs
                    return True
                
                x = row[-1] + 1
                while x < best_max:
                    if time.time() > deadline:
                        return False
                    
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    
                    if not conflict:
                        row.append(x)
                        if extend(pos + 1):
                            return True
                        row.pop()
                    
                    x += 1
                
                return False
            
            extend(1)
        
        backtrack(0)
    
    search()
    
    return best_rows
# EVOLVE-BLOCK-END
