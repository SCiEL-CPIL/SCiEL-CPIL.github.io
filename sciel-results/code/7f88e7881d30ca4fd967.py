import time
import random

def is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
        if sorted(sq[j][i] for j in range(n)) != list(range(n)):
            return False
    return True

def is_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 80
    
    # Base Latin square: cyclic construction
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second orthogonal Latin square using backtracking
    # with constraint propagation
    
    # For each cell (i, j), the value in sq2[i][j] must satisfy:
    # 1. It's not already used in row i of sq2
    # 2. It's not already used in column j of sq2
    # 3. The pair (base[i][j], sq2[i][j]) hasn't been used before
    
    def find_orthogonal(base_sq, time_limit):
        n = len(base_sq)
        sq = [[-1] * n for _ in range(n)]
        
        # Track constraints
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        # Precompute for each cell (i,j) the base value
        base_val = [[base_sq[i][j] for j in range(n)] for i in range(n)]
        
        # Order cells to fill: use a static ordering that helps pruning
        # Fill row by row, but we can also try different orderings
        cells = [(i, j) for i in range(n) for j in range(n)]
        
        def backtrack(idx):
            if time.time() > time_limit:
                return None
            
            if idx == n * n:
                return [row[:] for row in sq]
            
            i, j = cells[idx]
            bv = base_val[i][j]
            
            # MRV: try values that have fewest options remaining
            # For now, try all valid values
            candidates = []
            for v in range(n):
                if v not in row_used[i] and v not in col_used[j] and (bv, v) not in pair_used:
                    candidates.append(v)
            
            if not candidates:
                return None
            
            # Try each candidate
            for v in candidates:
                sq[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                pair_used.add((bv, v))
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                sq[i][j] = -1
                row_used[i].discard(v)
                col_used[j].discard(v)
                pair_used.discard((bv, v))
            
            return None
        
        return backtrack(0)
    
    # Try with a time limit
    time_limit = start_time + 75  # 5 seconds safety margin
    result = find_orthogonal(base, time_limit)
    
    if result is not None:
        # Verify
        if is_latin(result) and is_orthogonal(base, result):
            return [base, result]
    
    # If we couldn't find one, return just the base
    return [base]
