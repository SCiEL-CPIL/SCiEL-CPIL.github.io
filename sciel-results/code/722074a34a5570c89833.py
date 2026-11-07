import time
import sys

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    deadline = time.time() + 55
    
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    # Pre-compute available values for each cell?
    # Dynamic MRV is better.
    
    def get_mrv_cell():
        min_count = n + 1
        best_cell = None
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    count = 0
                    for val in range(n):
                        if val not in row_used[i] and val not in col_used[j] and (L1[i][j], val) not in pair_used:
                            count += 1
                    if count < min_count:
                        min_count = count
                        best_cell = (i, j)
                    if count == 0:
                        return (i, j, 0) # Fail fast
        return best_cell, min_count

    def backtrack():
        if time.time() > deadline:
            return False
            
        cell, count = get_mrv_cell()
        if cell is None:
            return True # All filled
        
        i, j = cell
        if count == 0:
            return False
        
        # Try values
        # Order: maybe try values that appear least in other constraints?
        # For now, 0..9
        for val in range(n):
            if val not in row_used[i] and val not in col_used[j] and (L1[i][j], val) not in pair_used:
                L2[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                pair_used.add((L1[i][j], val))
                
                if backtrack():
                    return True
                else:
                    L2[i][j] = -1
                    row_used[i].remove(val)
                    col_used[j].remove(val)
                    pair_used.remove((L1[i][j], val))
        
        return False

    found = backtrack()
    
    if found:
        # Verify
        for i in range(n):
            if sorted(L2[i]) != list(range(n)):
                return [L1]
            if sorted([L2[i][j] for i in range(n)]) != list(range(n)):
                return [L1]
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (L1[i][j], L2[i][j])
                if p in pairs:
                    return [L1]
                pairs.add(p)
        return [L1, L2]
    else:
        return [L1]
