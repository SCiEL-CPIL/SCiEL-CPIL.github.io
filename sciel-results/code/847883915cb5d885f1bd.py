import time

def construct():
    n = 10
    A = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    B = [[-1 for _ in range(n)] for _ in range(n)]
    
    row_used = [0] * n
    col_used = [0] * n
    diag_used = [0] * n
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds
    
    def solve(idx):
        nonlocal deadline
        if time.time() > deadline:
            return False
        
        if idx == n * n:
            return True
        
        i = idx // n
        j = idx % n
        k = (i + j) % n
        
        for val in range(n):
            bit = 1 << val
            if (row_used[i] & bit) or (col_used[j] & bit) or (diag_used[k] & bit):
                continue
            
            B[i][j] = val
            row_used[i] |= bit
            col_used[j] |= bit
            diag_used[k] |= bit
            
            if solve(idx + 1):
                return True
            
            B[i][j] = -1
            row_used[i] ^= bit
            col_used[j] ^= bit
            diag_used[k] ^= bit
            
        return False

    if solve(0):
        return [A, B]
    else:
        return [A]
