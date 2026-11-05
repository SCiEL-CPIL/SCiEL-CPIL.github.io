import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Decompose 4x5x5 into:
    # 1. 4x4x4 (top-left 4x4 of A, top-left 4x4 of B)
    # 2. 4x4x1 (top-left 4x4 of A, top-right 4x1 of B)
    # 3. 4x1x4 (bottom-right 4x1 of A, bottom-left 1x4 of B)
    # 4. 4x1x1 (bottom-right 4x1 of A, bottom-right 1x1 of B)
    
    # Indices:
    # A is 4x5. A[i, j] with i in 0..3, j in 0..4.
    # B is 5x5. B[j, k] with j in 0..4, k in 0..4.
    # C is 4x5. C[i, k] with i in 0..3, k in 0..4.
    
    # Split A into A_L (4x4) and A_R (4x1)
    # Split B into B_T (4x5) and B_B (1x5)
    # B_T is split into B_TL (4x4) and B_TR (4x1)
    # B_B is split into B_BL (1x4) and B_BR (1x1)
    
    # Term 1: A_L * B_TL (4x4x4)
    # Term 2: A_L * B_TR (4x4x1)
    # Term 3: A_R * B_BL (4x1x4)
    # Term 4: A_R * B_BR (4x1x1)
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(U_vals, V_vals, W_vals):
        nonlocal rank
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pass # Not iterating like this
        
        # We need to construct U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
        # But we are building them incrementally.
        pass

    # Let's just build the full matrices.
    # Max rank is 19.
    max_rank = 19
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # Term 1: 4x4x4 Schoolbook
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    # Term 2: 4x4x1
    # A_L (i, j) with i in 0..3, j in 0..3
    # B_TR (j, k) with j in 0..3, k=4
    # C (i, k) with i in 0..3, k=4
    # Rank 1
    # U[i*m+j, col] = 1 for i in 0..3, j in 0..3
    # V[j*p+k, col] = 1 for j in 0..3, k=4
    # W[k*n+i, col] = 1 for k=4, i in 0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1.0
    for j in range(4):
        V[j * p + 4][col] = 1.0
    for i in range(4):
        W[4 * n + i][col] = 1.0
    col += 1
    
    # Term 3: 4x1x4
    # A_R (i, j) with i in 0..3, j=4
    # B_BL (j, k) with j=4, k in 0..3
    # C (i, k) with i in 0..3, k in 0..3
    # Rank 1
    for i in range(4):
        U[i * m + 4][col] = 1.0
    for k in range(4):
        V[4 * p + k][col] = 1.0
    for i in range(4):
        for k in range(4):
            W[k * n + i][col] = 1.0
    col += 1
    
    # Term 4: 4x1x1
    # A_R (i, 4)
    # B_BR (4, 4)
    # C (i, 4)
    # Rank 1
    for i in range(4):
        U[i * m + 4][col] = 1.0
    V[4 * p + 4][col] = 1.0
    for i in range(4):
        W[4 * n + i][col] = 1.0
    col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
