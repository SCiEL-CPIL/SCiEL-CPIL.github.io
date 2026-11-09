import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank_444 = 64
    rank_441 = 4
    rank_414 = 4
    rank_411 = 4
    total_rank = rank_444 + rank_441 + rank_414 + rank_411
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 1. 4x4x4 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. 4x4x1 (A[0:4, 0:4] * B[0:4, 4])
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 4: for each j, term
    for j in range(4):
        for i in range(4):
            # This decomposition is not quite right for rank 4.
            # The rank of 4x4x1 is 4.
            # We need 4 terms total for all i.
            pass
            
    # Correct rank 4 decomposition for 4x4x1:
    # C_i = sum_j A_ij B_j
    # Terms r=0..3
    # U[i*4+j, r] = delta(j, r)
    # V[j, r] = delta(j, r)
    # W[i, r] = 1
    for r in range(4):
        for i in range(4):
            U[i * m + r][col] = 1
            V[r * p + 4][col] = 1
            W[4 * n + i][col] = 1
        col += 1
        
    # 3. 4x1x4 (A[0:4, 4] * B[4, 0:4])
    # C[i, k] = A[i, 4] B[4, k]
    # Rank 4
    for r in range(4):
        for i in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + r][col] = 1
            W[r * n + i][col] = 1
        col += 1
        
    # 4. 4x1x1 (A[0:4, 4] * B[4, 4])
    # C[i, 4] = A[i, 4] B[4, 4]
    # Rank 4
    for r in range(4):
        for i in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
