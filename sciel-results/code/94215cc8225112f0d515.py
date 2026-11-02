import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    # 2 blocks of 2x2x7, each rank 25
    # Each 2x2x7 is 3x(2x2x2) + 1x(2x2x1)
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_idx: indices for A (i, j)
    # b_idx: indices for B (j, k)
    # c_idx: indices for C (i, k)
    def add_strassen(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # M1 = (A[a_i, a_j] + A[a_i+1, a_j+1]) * (B[b_j, b_k] + B[b_j+1, b_k+1])
        # We need to represent this as U, V, W products.
        # U[row, col] * V[row, col] * W[row, col]
        
        # M1
        U[a_i * m + a_j][col] += 1
        U[(a_i+1) * m + (a_j+1)][col] += 1
        V[b_j * p + b_k][col] += 1
        V[(b_j+1) * p + (b_k+1)][col] += 1
        W[b_k * n + c_i][col] += 1
        W[(b_k+1) * n + (c_i+1)][col] += 1
        col += 1
        
        # M2 = (A[a_i+1, a_j] + A[a_i+1, a_j+1]) * B[b_j, b_k]
        U[(a_i+1) * m + a_j][col] += 1
        U[(a_i+1) * m + (a_j+1)][col] += 1
        V[b_j * p + b_k][col] += 1
        W[b_k * n + (c_i+1)][col] += 1
        col += 1
        
        # M3 = A[a_i, a_j] * (B[b_j+1, b_k] - B[b_j, b_k+1])
        U[a_i * m + a_j][col] += 1
        V[(b_j+1) * p + b_k][col] += 1
        V[b_j * p + (b_k+1)][col] -= 1
        W[b_k * n + c_i][col] += 1
        W[(b_k+1) * n + (c_i+1)][col] += 1
        col += 1
        
        # M4 = A[a_i+1, a_j+1] * (B[b_j+1, b_k] - B[b_j, b_k+1]) 
        # Wait, standard Strassen M4 = d(g-e). 
        # d = A[1,1] (local), g = B[1,0] (local), e = B[0,0] (local).
        # So d * (g - e).
        U[(a_i+1) * m + (a_j+1)][col] += 1
        V[(b_j+1) * p + b_k][col] += 1
        V[b_j * p + b_k][col] -= 1
        W[b_k * n + (c_i+1)][col] += 1
        col += 1
        
        # M5 = (A[a_i, a_j] + A[a_i, a_j+1]) * B[b_j+1, b_k+1]
        U[a_i * m + a_j][col] += 1
        U[a_i * m + (a_j+1)][col] += 1
        V[(b_j+1) * p + (b_k+1)][col] += 1
        W[(b_k+1) * n + c_i][col] += 1
        col += 1
        
        # M6 = (A[a_i+1, a_j] - A[a_i, a_j]) * (B[b_j, b_k] + B[b_j+1, b_k+1])
        U[(a_i+1) * m + a_j][col] += 1
        U[a_i * m + a_j][col] -= 1
        V[b_j * p + b_k][col] += 1
        V[(b_j+1) * p + (b_k+1)][col] += 1
        W[b_k * n + (c_i+1)][col] += 1
        W[(b_k+1) * n + (c_i+1)][col] += 1
        col += 1
        
        # M7 = (A[a_i, a_j+1] - A[a_i+1, a_j+1]) * (B[b_j+1, b_k] + B[b_j+1, b_k+1])
        U[a_i * m + (a_j+1)][col] += 1
        U[(a_i+1) * m + (a_j+1)][col] -= 1
        V[(b_j+1) * p + b_k][col] += 1
        V[(b_j+1) * p + (b_k+1)][col] += 1
        W[b_k * n + c_i][col] += 1
        W[(b_k+1) * n + c_i][col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # C_0 = A_00 B_0 + A_01 B_1
        U[a_i * m + a_j][col] += 1
        V[b_j * p + b_k][col] += 1
        W[b_k * n + c_i][col] += 1
        col += 1
        
        U[a_i * m + (a_j+1)][col] += 1
        V[(b_j+1) * p + b_k][col] += 1
        W[b_k * n + c_i][col] += 1
        col += 1
        
        # C_1 = A_10 B_0 + A_11 B_1
        U[(a_i+1) * m + a_j][col] += 1
        V[b_j * p + b_k][col] += 1
        W[b_k * n + (c_i+1)][col] += 1
        col += 1
        
        U[(a_i+1) * m + (a_j+1)][col] += 1
        V[(b_j+1) * p + b_k][col] += 1
        W[b_k * n + (c_i+1)][col] += 1
        col += 1

    # Part 1: A_1 (cols 0,1), B_1 (rows 0,1)
    # a_i=0, a_j=0
    # b_j=0
    # c_i=0
    
    # Block 1: k=0,1 (2x2x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    # Block 2: k=2,3 (2x2x2)
    add_strassen(0, 0, 0, 2, 0, 2)
    # Block 3: k=4,5 (2x2x2)
    add_strassen(0, 0, 0, 4, 0, 4)
    # Block 4: k=6 (2x2x1)
    add_schoolbook_2x2x1(0, 0, 0, 6, 0, 6)

    # Part 2: A_2 (cols 2,3), B_2 (rows 2,3)
    # a_i=0, a_j=2
    # b_j=2
    # c_i=0
    
    # Block 1: k=0,1
    add_strassen(0, 2, 2, 0, 0, 0)
    # Block 2: k=2,3
    add_strassen(0, 2, 2, 2, 0, 2)
    # Block 3: k=4,5
    add_strassen(0, 2, 2, 4, 0, 4)
    # Block 4: k=6
    add_schoolbook_2x2x1(0, 2, 2, 6, 0, 6)

    return {"U": U, "V": V, "W": W}
