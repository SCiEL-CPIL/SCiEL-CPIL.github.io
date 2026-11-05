import time
import random

def construct(n_rows, k):
    def greedy_solution():
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows
    
    best_solution = greedy_solution()
    best_max = max(max(r) for r in best_solution)
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    current_solution = [list(r) for r in best_solution]
    current_max = best_max
    
    start_time = time.time()
    time_limit = 115
    
    while time.time() - start_time < time_limit:
        improved = False
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for row_idx in row_order:
            if time.time() - start_time >= time_limit:
                break
            
            other_diffs = set()
            for i in range(n_rows):
                if i == row_idx:
                    continue
                row = current_solution[i]
                for a in range(len(row)):
                    for b in range(a+1, len(row)):
                        other_diffs.add(row[b] - row[a])
            
            current_row = current_solution[row_idx]
            current_row_max = current_row[-1]
            
            for target_max in range(current_row_max - 1, max(0, current_row_max - 40), -1):
                if time.time() - start_time >= time_limit:
                    break
                
                node_count = [0]
                max_nodes = 50000
                
                def dfs(pos, curr, used):
                    if node_count[0] > max_nodes:
                        return None
                    node_count[0] += 1
                    
                    if pos == k + 1:
                        return list(curr)
                    if pos == 0:
                        return dfs(1, [0], used)
                    start = curr[-1] + 1
                    end = min(target_max, start + 30)
                    for x in range(start, end + 1):
                        new_diffs = {x - a for a in curr}
                        if new_diffs & used or new_diffs & other_diffs:
                            continue
                        found = dfs(pos + 1, curr + [x], used | new_diffs)
                        if found:
                            return found
                    return None
                
                found_row = dfs(0, [], set())
                if found_row:
                    current_solution[row_idx] = found_row
                    current_max = max(max(r) for r in current_solution)
                    if current_max < best_max:
                        best_max = current_max
                        best_solution = [list(r) for r in current_solution]
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_solution
