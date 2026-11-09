import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using a backtracking search on S(4,5,11).
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # S(4,5,11): 66 blocks, each 5-subset of {0..10}, covering each 4-subset exactly once
    points = list(range(11))
    all_4subs = list(combinations(points, 4))
    col_idx = {s: i for i, s in enumerate(all_4subs)}
    n_cols = len(all_4subs)  # 330
    
    all_5subs = list(combinations(points, 5))
    # For each 5-subset, compute which 4-subsets it contains
    row_cols = []
    for b in all_5subs:
        cols = [col_idx[s] for s in combinations(b, 4)]
        row_cols.append(cols)
    
    # Build col -> rows mapping
    col_to_rows = [[] for _ in range(n_cols)]
    for r, cols in enumerate(row_cols):
        for c in cols:
            col_to_rows[c].append(r)
    
    full_mask = (1 << n_cols) - 1
    solution = []
    used_rows = 0
    covered = 0
    
    def search():
        nonlocal used_rows, covered, solution
        
        if time.time() > deadline:
            return False
        
        if covered == full_mask:
            return True
        
        # MRV: find uncovered column with fewest candidates
        min_count = float('inf')
        best_col = -1
        best_cands = []
        
        for c in range(n_cols):
            if (covered >> c) & 1:
                continue
            cands = []
            for r in col_to_rows[c]:
                if (used_rows >> r) & 1:
                    continue
                # Check no conflict with other covered cols
                conflict = False
                for cc in row_cols[r]:
                    if cc != c and (covered >> cc) & 1:
                        conflict = True
                        break
                if not conflict:
                    cands.append(r)
            if len(cands) < min_count:
                min_count = len(cands)
                best_col = c
                best_cands = cands
                if min_count == 0:
                    return False
                if min_count == 1:
                    break
        
        if best_col == -1:
            return False
        
        for r in best_cands:
            used_rows |= (1 << r)
            new_bits = 0
            for cc in row_cols[r]:
                if not ((covered >> cc) & 1):
                    new_bits |= (1 << cc)
            covered |= new_bits
            solution.append(r)
            
            if search():
                return True
            
            solution.pop()
            used_rows &= ~(1 << r)
            covered &= ~new_bits
        
        return False
    
    import sys
    sys.setrecursionlimit(10000)
    
    if not search():
        return []
    
    # Build S(5,6,12) from S(4,5,11)
    result = []
    V = set(range(11))
    for r in solution:
        B = set(all_5subs[r])
        result.append(sorted(list(B) + [11]))
        comp = V - B
        result.append(sorted(list(comp)))
    
    return result
