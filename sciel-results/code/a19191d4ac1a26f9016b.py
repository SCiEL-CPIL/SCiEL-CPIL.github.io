import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe with 90s limit
    n = 10
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    used_pairs = set()
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Fix first row of L2 to 0..9 due to symbol permutation symmetry
    for j in range(n):
        L2[0][j] = j
        row_used[0].add(j)
        col_used[j].add(j)
        used_pairs.add((L1[0][j], j))
        
    def backtrack(i, j):
        if time.time() > deadline:
            return False
            
        if i == n:
            return True
            
        if j == n:
            return backtrack(i + 1, 0)
            
        if L2[i][j] != -1:
            return backtrack(i, j + 1)
            
        l1_val = L1[i][j]
        for val in range(n):
            if val in row_used[i]:
                continue
            if val in col_used[j]:
                continue
            pair = (l1_val, val)
            if pair in used_pairs:
                continue
                
            L2[i][j] = val
            row_used[i].add(val)
            col_used[j].add(val)
            used_pairs.add(pair)
            
            if backtrack(i, j + 1):
                return True
            
            L2[i][j] = -1
            row_used[i].remove(val)
            col_used[j].remove(val)
            used_pairs.remove(pair)
            
        return False

    found = backtrack(1, 0)
    
    squares = [L1]
    if found:
        squares.append(L2)
        
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
