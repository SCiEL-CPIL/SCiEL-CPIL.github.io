import time
import random

def construct():
    n = 10
    deadline = time.time() + 80
    
    def is_latin_square(sq):
        for i in range(n):
            row = [int(x) for x in sq[i]]
            if len(set(row)) != n:
                return False
        for j in range(n):
            col = [int(sq[i][j]) for i in range(n)]
            if len(set(col)) != n:
                return False
        return True
    
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (int(sq1[i][j]), int(sq2[i][j]))
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True
    
    # L1: standard addition table mod 10
    L1 = [[int((i + j) % n) for j in range(n)] for i in range(n)]
    
    def find_orthogonal(L1):
        cells_by_L1 = {s: [] for s in range(n)}
        for i in range(n):
            for j in range(n):
                cells_by_L1[L1[i][j]].append((i, j))
        
        L2 = [[None] * n for _ in range(n)]
        
        def is_valid(i, j, val):
            for j2 in range(n):
                if j2 != j and L2[i][j2] == val:
                    return False
            for i2 in range(n):
                if i2 != i and L2[i2][j] == val:
                    return False
            for (i2, j2) in cells_by_L1[L1[i][j]]:
                if (i2, j2) != (i, j) and L2[i2][j2] == val:
                    return False
            return True
        
        cells = [(i, j) for i in range(n) for j in range(n)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n * n:
                return True
            
            i, j = cells[idx]
            
            vals = list(range(n))
            random.shuffle(vals)
            
            for val in vals:
                if is_valid(i, j, val):
                    L2[i][j] = val
                    if backtrack(idx + 1):
                        return True
                    L2[i][j] = None
            
            return False
        
        if backtrack(0):
            return [[int(L2[i][j]) for j in range(n)] for i in range(n)]
        return None
    
    L2 = find_orthogonal(L1)
    
    if L2 is not None:
        if is_latin_square(L1) and is_latin_square(L2) and are_orthogonal(L1, L2):
            return [L1, L2]
    
    return [L1]
