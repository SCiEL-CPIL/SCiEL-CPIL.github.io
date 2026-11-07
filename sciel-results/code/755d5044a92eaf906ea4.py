import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_square(orthogonal_to):
        # orthogonal_to is a list of squares
        grid = [[0]*n for _ in range(n)]
        row_masks = [0]*n
        col_masks = [0]*n
        
        # pair_used[k][a][b] is True if pair (a, b) has been used for square k
        # a is from orthogonal_to[k], b is from new square
        pair_used = []
        for _ in orthogonal_to:
            pair_used.append([[False]*n for _ in range(n)])
        
        # Normalize first row to identity
        for j in range(n):
            grid[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            for k, A_orth in enumerate(orthogonal_to):
                pair_used[k][A_orth[0][j]][j] = True
            
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            
            next_i, next_j = (i+1, 0) if j == n-1 else (i, j+1)
            
            for s in range(n):
                bit = 1 << s
                if (row_masks[i] & bit) or (col_masks[j] & bit):
                    continue
                
                # Check orthogonality
                valid = True
                for k, A_orth in enumerate(orthogonal_to):
                    if pair_used[k][A_orth[i][j]][s]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                for k, A_orth in enumerate(orthogonal_to):
                    pair_used[k][A_orth[i][j]][s] = True
                
                if backtrack(next_i, next_j):
                    return True
                
                # Unplace
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                for k, A_orth in enumerate(orthogonal_to):
                    pair_used[k][A_orth[i][j]][s] = False
                
            return False
        
        if backtrack(1, 0):
            return grid
        return None

    # Find A1
    A1 = find_square([A0])
    if A1 is not None:
        squares.append(A1)
        
    # Find A2 orthogonal to A0 and A1
    if time.time() < deadline - 5.0: # Safety margin
        A2 = find_square([A0, A1])
        if A2 is not None:
            squares.append(A2)
            
    return squares
