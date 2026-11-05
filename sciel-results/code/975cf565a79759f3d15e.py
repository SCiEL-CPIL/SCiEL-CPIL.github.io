import sys
import time
from itertools import combinations

def construct():
    sys.setrecursionlimit(2000)
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    n = 12
    points = range(n)
    
    all_5_subsets = list(combinations(points, 5))
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets)
    
    all_6_subsets = list(combinations(points, 6))
    num_6_subsets = len(all_6_subsets)
    
    row_cols = [None] * num_6_subsets
    for r, block in enumerate(all_6_subsets):
        cols = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            cols.append(subset_to_id[sub])
        row_cols[r] = cols
        
    column_rows = [set() for _ in range(num_5_subsets)]
    for r in range(num_6_subsets):
        for c in row_cols[r]:
            column_rows[c].add(r)
            
    initial_available = set(range(num_5_subsets))
    best_solution = []
    
    def solve(available_cols, selected_rows):
        nonlocal best_solution, deadline, start_time
        
        if time.time() - start_time > deadline:
            return False
            
        if not available_cols:
            if len(selected_rows) > len(best_solution):
                best_solution = selected_rows.copy()
            return True
            
        min_count = float('inf')
        best_col = -1
        best_rows = []
        
        for c in available_cols:
            valid_rows = []
            for r in column_rows[c]:
                if all(cc in available_cols for cc in row_cols[r]):
                    valid_rows.append(r)
                    
            if len(valid_rows) == 0:
                return False
                
            if len(valid_rows) < min_count:
                min_count = len(valid_rows)
                best_col = c
                best_rows = valid_rows
                if min_count == 1:
                    break
        
        for r in best_rows:
            new_available = available_cols - set(row_cols[r])
            if solve(new_available, selected_rows + [r]):
                return True
                
        return False

    solve(initial_available, [])
    
    if best_solution:
        return [list(all_6_subsets[r]) for r in best_solution]
    else:
        return []
