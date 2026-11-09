import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Base Latin Square
    # L0[i][j] = (i + j) % n
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [base]
    
    # Backtracking solver to find the next orthogonal square
    def find_next_square(existing_squares):
        num_existing = len(existing_squares)
        
        grid = [[-1] * n for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        # used_pairs[k][p] is True if pair p is used for existing square k
        # p = val_existing * n + val_new
        used_pairs = [[False] * (n * n) for _ in range(num_existing)]
        
        def backtrack(idx):
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            avail_mask = 0
            for s in range(n):
                if (row_masks[i] & (1 << s)) == 0 and (col_masks[j] & (1 << s)) == 0:
                    ok = True
                    for k in range(num_existing):
                        p = existing_squares[k][i][j] * n + s
                        if used_pairs[k][p]:
                            ok = False
                            break
                    if ok:
                        avail_mask |= (1 << s)
            
            if avail_mask == 0:
                return False
            
            symbols = []
            for s in range(n):
                if avail_mask & (1 << s):
                    symbols.append(s)
            
            random.shuffle(symbols)
            
            for s in symbols:
                grid[i][j] = s
                row_masks[i] |= (1 << s)
                col_masks[j] |= (1 << s)
                for k in range(num_existing):
                    p = existing_squares[k][i][j] * n + s
                    used_pairs[k][p] = True
                
                if backtrack(idx + 1):
                    return True
                
                row_masks[i] ^= (1 << s)
                col_masks[j] ^= (1 << s)
                for k in range(num_existing):
                    p = existing_squares[k][i][j] * n + s
                    used_pairs[k][p] = False
                grid[i][j] = -1
            
            return False
        
        if backtrack(0):
            return grid
        return None

    # Try to add squares
    while len(squares) < 9:
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        while attempts < 10: # Allow more retries
            if time.time() > deadline:
                break
            
            next_sq = find_next_square(squares)
            if next_sq is not None:
                squares.append(next_sq)
                found = True
                break
            attempts += 1
        
        if not found:
            break
            
    return squares
