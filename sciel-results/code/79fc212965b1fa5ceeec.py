DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import numpy as np
    
    # Strassen 2x2x2 rank 7 matrices
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    # Part 1: 4x4x4 multiplication (A * B1)
    # B1 is 4x4. C1 is 4x4.
    # Use 4 blocks of 2x2x2.
    rank_444 = 4 * 7 # 28
    
    # Part 2: 4x4x2 multiplication (A * B2)
    # B2 is 4x2. C2 is 4x2.
    # This is equivalent to two 4x4x1 multiplications.
    # Rank of 4x4x1 is 4.
    # So rank is 4 + 4 = 8.
    rank_442 = 8
    
    total_rank = rank_444 + rank_442 # 36
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Fill U, V, W for the 4x4x4 part (cols 0..27)
    # B1 corresponds to k=0..3
    # C1 corresponds to k=0..3 (in W, index k*n+i)
    col_offset = 0
    for bi in range(2): # i blocks
        for bj in range(2): # j blocks
            for bk in range(2): # k blocks (for B1, k is 0..3)
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j, col] = S_U[di * 2 + dj, c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k, col] = S_V[dj * 2 + dk, c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i, col] = S_W[dk * 2 + di, c]
                col_offset += 7
                
    # Fill U, V, W for the 4x4x2 part (cols 28..35)
    # B2 corresponds to k=4..5
    # C2 corresponds to k=4..5
    # We use rank-4 decomposition for each 4x4x1.
    # For M_4,4,1, T[i,j,k] = 1 for all i,j.
    # Decomposition: sum_{r=1}^4 e_r (in A) otimes 1 (in B) otimes e_r (in C)?
    # Let's check.
    # U_r[i,j] = delta_{i,r} (for j any? No, A is 4x4).
    # For M_4,4,1, B is 4x1.
    # U_r[i,j] = delta_{i,r} * 1?
    # V_r[j,k] = 1?
    # W_r[k,i] = delta_{i,r}?
    # Sum_r delta_{i,r} * 1 * delta_{i,r} = delta_{i,r} (if i=r) -> 1.
    # Wait, sum over r.
    # If i=r, term is 1. Else 0.
    # So sum is 1 if i is in 0..3.
    # This works for all i,j.
    
    # For k=4:
    for r in range(4):
        col = col_offset + r
        # U: i, j. U[i*m+j, col] = 1 if i == r, else 0.
        for i in range(n):
            for j in range(m):
                if i == r:
                    U[i * m + j, col] = 1.0
        # V: j, k. V[j*p+k, col] = 1 for all j, k=4.
        for j in range(m):
            V[j * p + 4, col] = 1.0
        # W: k, i. W[k*n+i, col] = 1 if i == r, else 0. (k=4)
        for i in range(n):
            if i == r:
                W[4 * n + i, col] = 1.0
    col_offset += 4
    
    # For k=5:
    for r in range(4):
        col = col_offset + r
        for i in range(n):
            for j in range(m):
                if i == r:
                    U[i * m + j, col] = 1.0
        for j in range(m):
            V[j * p + 5, col] = 1.0
        for i in range(n):
            if i == r:
                W[5 * n + i, col] = 1.0
    col_offset += 4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
