import time

def construct():
    n = 10
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Base Latin square: L1[i][j] = (i+j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Verify L1 is a valid Latin square
    for i in range(n):
        if len(set(L1[i])) != n:
            return [L1]
    for j in range(n):
        col = [L1[i][j] for i in range(n)]
        if len(set(col)) != n:
            return [L1]
    
    # We need to find L2 orthogonal to L1
    # Backtracking with constraint propagation
    L2 = [[-1] * n for _ in range(n)]
    
    # For each row, track which symbols are used
    row_symbols = [set() for _ in range(n)]
    # For each column, track which symbols are used
    col_symbols = [set() for _ in range(n)]
    
    # pair_used[symbol_in_L1][symbol_in_L2] = True if this pair has been used
    pair_used = [[False] * n for _ in range(n)]
    
    def is_valid(i, j, val):
        if val in row_symbols[i]:
            return False
        if val in col_symbols[j]:
            return False
        l1_val = L1[i][j]
        if pair_used[l1_val][val]:
            return False
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n * n:
            return True
        i = pos // n
        j = pos % n
        
        for val in range(n):
            if is_valid(i, j, val):
                L2[i][j] = val
                row_symbols[i].add(val)
                col_symbols[j].add(val)
                l1_val = L1[i][j]
                pair_used[l1_val][val] = True
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                L2[i][j] = -1
                row_symbols[i].remove(val)
                col_symbols[j].remove(val)
                pair_used[l1_val][val] = False
        
        return False
    
    # Try to find L2
    found = backtrack(0)
    
    if not found:
        return [L1]
    
    # Verify L2 is a valid Latin square
    for i in range(n):
        if len(set(L2[i])) != n:
            return [L1]
    for j in range(n):
        col = [L2[i][j] for i in range(n)]
        if len(set(col)) != n:
            return [L1]
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            if pair in pairs:
                return [L1]
            pairs.add(pair)
    
    result = [L1, L2]
    return result
