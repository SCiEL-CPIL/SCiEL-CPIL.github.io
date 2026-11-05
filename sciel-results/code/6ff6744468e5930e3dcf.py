"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import copy

def construct(n_rows, k):
    def greedy_construct():
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
        return rows
    
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    def get_max(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))
    
    def local_search(rows, iterations=20000):
        rows = copy.deepcopy(rows)
        current_max = get_max(rows)
        best_rows = copy.deepcopy(rows)
        best_max = current_max
        
        row_diffs = [get_row_diffs(row) for row in rows]
        
        temp = 0.5
        cooling = 0.9997
        
        for it in range(iterations):
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(rows[r]) - 1)
            old_val = rows[r][pos]
            
            min_val = rows[r][pos-1] + 1
            if min_val >= old_val:
                temp *= cooling
                continue
            
            new_val = random.randint(min_val, old_val)
            if new_val == old_val:
                continue
            
            old_diffs = row_diffs[r]
            
            removed_diffs = set()
            for i in range(pos):
                removed_diffs.add(old_val - rows[r][i])
            for j in range(pos+1, len(rows[r])):
                removed_diffs.add(rows[r][j] - old_val)
            
            rows[r][pos] = new_val
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - rows[r][i])
            for j in range(pos+1, len(rows[r])):
                new_diffs.add(rows[r][j] - new_val)
            
            all_row_diffs = (old_diffs - removed_diffs) | new_diffs
            if len(all_row_diffs) != len(set(all_row_diffs)):
                rows[r][pos] = old_val
                temp *= cooling
                continue
            
            conflict = False
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                if all_row_diffs & row_diffs[other_r]:
                    conflict = True
                    break
            
            if conflict:
                rows[r][pos] = old_val
                temp *= cooling
                continue
            
            new_max = max(max(row) for row in rows)
            delta = new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
                current_max = new_max
                row_diffs[r] = all_row_diffs
                if current_max < best_max:
                    best_max = current_max
                    best_rows = copy.deepcopy(rows)
            
            temp *= cooling
        
        return best_rows
    
    rows = greedy_construct()
    best_rows = rows
    best_max = get_max(rows)
    
    for restart in range(10):
        if restart == 0:
            search_rows = copy.deepcopy(best_rows)
        else:
            search_rows = copy.deepcopy(best_rows)
            for _ in range(random.randint(2, 5)):
                r1 = random.randint(0, n_rows - 1)
                r2 = random.randint(0, n_rows - 1)
                if r1 == r2:
                    continue
                p1 = random.randint(1, len(search_rows[r1]) - 1)
                p2 = random.randint(1, len(search_rows[r2]) - 1)
                v1, v2 = search_rows[r1][p1], search_rows[r2][p2]
                
                ok1 = True
                if p1 > 0 and v2 <= search_rows[r1][p1-1]:
                    ok1 = False
                if p1 < len(search_rows[r1]) - 1 and v2 >= search_rows[r1][p1+1]:
                    ok1 = False
                ok2 = True
                if p2 > 0 and v1 <= search_rows[r2][p2-1]:
                    ok2 = False
                if p2 < len(search_rows[r2]) - 1 and v1 >= search_rows[r2][p2+1]:
                    ok2 = False
                
                if ok1 and ok2:
                    search_rows[r1][p1], search_rows[r2][p2] = v2, v1
            
            if not is_valid(search_rows):
                search_rows = copy.deepcopy(best_rows)
        
        improved = local_search(search_rows, iterations=8000)
        if get_max(improved) < best_max:
            best_max = get_max(improved)
            best_rows = improved
    
    return best_rows
# EVOLVE-BLOCK-END
