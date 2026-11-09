import time
import sys
from itertools import combinations

def construct():
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_cols = len(all_5subsets) # 792
    
    col_index = {s: i for i, s in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(points, 6))
    
    row_masks = [0] * len(all_6subsets)
    col_to_rows = [[] for _ in range(num_cols)]
    
    for r_idx, b in enumerate(all_6subsets):
        mask = 0
        for s in combinations(b, 5):
            c_idx = col_index[s]
            mask |= (1 << c_idx)
            col_to_rows[c_idx].append(r_idx)
        row_masks[r_idx] = mask
        
    solution = []
    covered_mask = 0
    num_cols_mask = (1 << num_cols) - 1
    deadline = time.time() + 145.0 # 5s margin
    
    sys.setrecursionlimit(5000)
    
    def search():
        nonlocal covered_mask, solution
        
        if time.time() > deadline:
            raise TimeoutError()
            
        if covered_mask == num_cols_mask:
            return True
            
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        for c in range(num_cols):
            if (covered_mask >> c) & 1:
                continue
                
            candidates = []
            for r in col_to_rows[c]:
                if row_masks[r] & covered_mask == 0:
                    candidates.append(r)
            
            count = len(candidates)
            if count < min_count:
                min_count = count
                best_col = c
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                if min_count == 1:
                    break
            
        if best_col == -1:
            return False
            
        for r in best_candidates:
            solution.append(r)
            old_mask = covered_mask
            covered_mask |= row_masks[r]
            
            if search():
                return True
            
            covered_mask = old_mask
            solution.pop()
            
        return False

    try:
        if search():
            result = [list(all_6subsets[r_idx]) for r_idx in solution]
            return result
    except TimeoutError:
        pass
        
    return []

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
