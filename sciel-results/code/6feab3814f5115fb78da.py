import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Kronecker product: (3,4,8) = (3,2,4) ⊗ (1,2,2)
    # (3,2,4): multiply 3x2 by 2x4, decomposed as two (3,2,2) blocks
    # (1,2,2): multiply 1x2 by 2x2, rank 2
    
    # For (3,2,2): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Rank 6 decomposition:
    # U indices: i*2+j (i=0..2, j=0..1) -> 6 entries
    # V indices: j*2+k (j=0..1, k=0..1) -> 4 entries  
    # W indices: k*3+i (k=0..1, i=0..2) -> 6 entries
    
    # Actually, let me build the full (3,4,8) tensor directly with rank 24
    # using the Kronecker structure
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 12 x 24
    V = [[0] * rank for _ in range(m * p)]  # 32 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    # (3,2,4) part: for each of the 2 blocks in the 4-dimension
    # (1,2,2) part: for each of the 2 blocks in the 8-dimension
    
    # Let's index: original i in 0..2, j in 0..3, k in 0..7
    # Factor: i = i1 (0..2), j = j1*2 + j2 where j1 in 0..1, j2 in 0..1
    # k = k1*2 + k2 where k1 in 0..3, k2 in 0..1
    
    # C[i1, j1*2+j2, k1*2+k2] = sum over j1',k1' ... 
    # This is getting complex. Let me just use schoolbook with rank 24 by
    # grouping terms.
    
    # Simpler: just do schoolbook but only 24 terms by exploiting structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{3} A[i,j] * B[j,k]
    # We can't reduce below 4*3*8/... 
    
    # Let me just implement rank 24 via Kronecker properly
    # (3,2,4) rank 12, (1,2,2) rank 2, total 24
    
    col = 0
    for block2 in range(2):  # (1,2,2) second factor: k2 dimension
        for block1 in range(2):  # (3,2,4) first factor: k1 block
            for j1 in range(2):  # (3,2,2) middle dimension
                for i1 in range(3):  # (3,2,2) first dimension
                    # U index: i1*4 + (block1*2 + j1) = i1*4 + j
                    u_idx = i1 * 4 + (block1 * 2 + j1)
                    # V index: (block1*2 + j1)*4 + (block2*2 + 0) ... 
                    # This is getting too complex without clear structure
                    
                    U[u_idx][col] = 1
                    col += 1
    
    # This approach is error-prone. Let me just return the schoolbook rank 96
    # and accept that the Kronecker approach needs more careful implementation.
    
    # For now, return schoolbook (same as before)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
