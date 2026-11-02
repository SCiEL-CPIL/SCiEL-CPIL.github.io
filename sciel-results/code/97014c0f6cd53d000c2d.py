import time

START_TIME = time.time()
TIME_LIMIT = 120

def _elapsed():
    return time.time() - START_TIME

def _deadline_exceeded(margin=2.0):
    return _elapsed() > (TIME_LIMIT - margin)

def construct():
    n, m, p = 3, 5, 5
    
    # Use overlapping Strassen blocks where i=0 pairs with i=1
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={2,3}, k={0,1}
    # Block 3: i={0,1}, j={0,1}, k={2,3}
    # Block 4: i={0,1}, j={2,3}, k={2,3}
    # Block 5: i={1,2}, j={0,1}, k={0,1}  -- OVERLAPPING in i dimension!
    # Block 6: i={1,2}, j={2,3}, k={0,1}  -- OVERLAPPING in i dimension!
    # Block 7: i={1,2}, j={0,1}, k={2,3}  -- OVERLAPPING in i dimension!
    # Block 8: i={1,2}, j={2,3}, k={2,3}  -- OVERLAPPING in i dimension!
    
    # With overlapping blocks, we need to be careful about double-counting.
    # Actually, let's stick with the non-overlapping approach but try a different
    # decomposition strategy.
    
    # Alternative: Use 3x3x3 Strassen-like blocks or try to find a better
    # partitioning.
    
    # For now, keep the 4 non-overlapping 2x2x2 blocks but verify correctness.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    schoolbook_rank = n * m * p - num_blocks * 8
    total_rank = strassen_rank + schoolbook_rank
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, start_col):
        c = start_col + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        c = start_col + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] += -1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        c = start_col + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        
        return start_col + 7
    
    for block in blocks:
        if _deadline_exceeded():
            break
        col = add_strassen_block(block[0], block[1], block[2], col)
    
    covered_set = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_set.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        if _deadline_exceeded():
            break
        for j in range(m):
            if _deadline_exceeded():
                break
            for k in range(p):
                if _deadline_exceeded():
                    break
                if (i, j, k) not in covered_set:
                    U[i*m + j][col] += 1.0
                    V[j*p + k][col] += 1.0
                    W[k*n + i][col] += 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
