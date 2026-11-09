import sys
import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) by solving the exact cover for S(4,5,11)
    using an optimized backtracking search with pre-computed bitmasks.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Solve for S(4,5,11)
    points_11 = range(11)
    all_4subsets = list(combinations(points_11, 4))
    col_index_4 = {s: i for i, s in enumerate(all_4subsets)}
    num_cols_4 = len(all_4subsets)  # 330
    
    all_5subsets_11 = list(combinations(points_11, 5))
    num_rows_11 = len(all_5subsets_11)  # 462
    
    # Pre-compute row masks and column lists
    rows_11 = []
    row_masks_11 = []
    for b in all_5subsets_11:
        cols = [col_index_4[s] for s in combinations(b, 4)]
        rows_11.append(cols)
        mask = 0
        for c in cols:
            mask |= (1 << c)
        row_masks_11.append(mask)
        
    col_to_rows_4 = [[] for _ in range(num_cols_4)]
    for r_idx, cols in enumerate(rows_11):
        for c in cols:
            col_to_rows_4[c].append(r_idx)
            
    solution_11 = []
    used_rows_11_mask = 0
    covered_4_mask = 0
    full_cover_mask = (1 << num_cols_4) - 1
    
    # Pre-compute candidate lists for each column to avoid re-filtering used rows repeatedly
    # But used_rows changes, so we still need to check.
    # However, we can pre-compute the list of rows for each col.
    
    def search_11(depth):
        nonlocal used_rows_11_mask, covered_4_mask, solution_11
        
        if time.time() > deadline:
            return False
            
        if covered_4_mask == full_cover_mask:
            return True
            
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        # Find the column with the fewest candidates (MRV heuristic)
        for c in range(num_cols_4):
            if (covered_4_mask >> c) & 1:
                continue
            
            candidates = []
            for r in col_to_rows_4[c]:
                if (used_rows_11_mask >> r) & 1:
                    continue
                # Check conflict: does row r cover any other covered col?
                # row_masks_11[r] & covered_4_mask should only have bit c set
                # i.e., (row_masks_11[r] & covered_4_mask) == (1 << c)
                if (row_masks_11[r] & covered_4_mask) == (1 << c):
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
            used_rows_11_mask |= (1 << r)
            
            # Update covered mask
            new_covered = row_masks_11[r] & ~covered_4_mask
            covered_4_mask |= new_covered
            
            solution_11.append(r)
            
            if search_11(depth + 1):
                return True
            
            solution_11.pop()
            used_rows_11_mask &= ~(1 << r)
            covered_4_mask &= ~new_covered
                
        return False

    sys.setrecursionlimit(10000)
    
    if not search_11(0):
        return []

    # 2. Construct S(5,6,12) from S(4,5,11)
    result = []
    V = set(range(11))
    for r_idx in solution_11:
        B = set(all_5subsets_11[r_idx])
        # Block 1: B U {11}
        result.append(sorted(list(B) + [11]))
        # Block 2: V \ B
        comp = V - B
        result.append(sorted(list(comp)))
        
    return result
